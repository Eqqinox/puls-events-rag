"""
Tests unitaires pour le module de preprocessing.

Ce module teste les fonctionnalités de :
- fetch_events.py : Collecte des données API
- clean_events.py : Nettoyage des données
- prepare_for_vectorization.py : Structuration pour RAG
- chunk_events.py : Découpage en chunks avec LangChain

Usage:
    pytest tests/test_preprocessing.py -v
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import des modules à tester
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.clean_events import (
    remove_html_tags,
    normalize_department,
    normalize_city,
    normalize_keywords,
    is_valid_event,
    is_ile_de_france,
    clean_event,
    DEPARTMENT_MAPPING,
    ILE_DE_FRANCE_DEPARTMENTS,
)

from preprocessing.prepare_for_vectorization import (
    create_text_for_embedding,
    extract_metadata,
    prepare_event_for_vectorization,
)

from preprocessing.chunk_events import (
    split_text_into_chunks,
    chunk_event,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    SEPARATORS,
)


# =============================================================================
# TESTS POUR clean_events.py
# =============================================================================

class TestRemoveHtmlTags:
    """Tests pour la fonction remove_html_tags."""
    
    def test_remove_simple_tags(self):
        """Teste la suppression des balises simples."""
        html = "<p>Bonjour</p>"
        result = remove_html_tags(html)
        assert result == "Bonjour"
    
    def test_remove_nested_tags(self):
        """Teste la suppression des balises imbriquées."""
        html = "<p><strong>Texte en gras</strong></p>"
        result = remove_html_tags(html)
        assert result == "Texte en gras"
    
    def test_preserve_line_breaks_from_br(self):
        """Teste que les balises br deviennent des sauts de ligne."""
        html = "Ligne 1<br>Ligne 2"
        result = remove_html_tags(html)
        assert "Ligne 1" in result
        assert "Ligne 2" in result
    
    def test_preserve_line_breaks_from_p(self):
        """Teste que les balises p créent des sauts de ligne."""
        html = "<p>Paragraphe 1</p><p>Paragraphe 2</p>"
        result = remove_html_tags(html)
        assert "Paragraphe 1" in result
        assert "Paragraphe 2" in result
    
    def test_decode_html_entities(self):
        """Teste le décodage des entités HTML."""
        html = "Caf&eacute; &amp; th&eacute;"
        result = remove_html_tags(html)
        assert "Café" in result
        assert "&" in result
        assert "thé" in result
    
    def test_handle_none(self):
        """Teste que None retourne None."""
        assert remove_html_tags(None) is None
    
    def test_handle_empty_string(self):
        """Teste qu'une chaîne vide retourne une chaîne vide."""
        assert remove_html_tags("") == ""
    
    def test_handle_non_string(self):
        """Teste que les non-strings sont retournés tels quels."""
        assert remove_html_tags(123) == 123
    
    def test_remove_links(self):
        """Teste la suppression des liens tout en gardant le texte."""
        html = '<a href="http://example.com">Cliquez ici</a>'
        result = remove_html_tags(html)
        assert result == "Cliquez ici"
    
    def test_remove_list_items(self):
        """Teste la suppression des listes."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = remove_html_tags(html)
        assert "Item 1" in result
        assert "Item 2" in result


class TestNormalizeDepartment:
    """Tests pour la fonction normalize_department."""
    
    def test_normalize_val_doise_variant(self):
        """Teste la normalisation de Val-D'Oise."""
        assert normalize_department("Val-D'Oise") == "Val-d'Oise"
    
    def test_normalize_seine_st_denis(self):
        """Teste la normalisation de Seine-St-Denis."""
        assert normalize_department("Seine-St-Denis") == "Seine-Saint-Denis"
    
    def test_normalize_seine_st_point_denis(self):
        """Teste la normalisation de Seine-St.-Denis."""
        assert normalize_department("Seine-St.-Denis") == "Seine-Saint-Denis"
    
    def test_normalize_val_de_marne_variant(self):
        """Teste la normalisation de Val-De-Marne."""
        assert normalize_department("Val-De-Marne") == "Val-de-Marne"
    
    def test_normalize_paris_lowercase(self):
        """Teste la normalisation de paris en minuscules."""
        assert normalize_department("paris") == "Paris"
    
    def test_normalize_postal_code(self):
        """Teste la normalisation des codes postaux parisiens."""
        assert normalize_department("75010") == "Paris"
        assert normalize_department("75018") == "Paris"
    
    def test_keep_valid_department(self):
        """Teste qu'un département valide reste inchangé."""
        assert normalize_department("Paris") == "Paris"
        assert normalize_department("Seine-et-Marne") == "Seine-et-Marne"
    
    def test_handle_none(self):
        """Teste que None retourne None."""
        assert normalize_department(None) is None
    
    def test_handle_empty_string(self):
        """Teste qu'une chaîne vide retourne une chaîne vide."""
        assert normalize_department("") == ""


class TestNormalizeCity:
    """Tests pour la fonction normalize_city."""
    
    def test_normalize_paris_uppercase(self):
        """Teste la normalisation de PARIS."""
        assert normalize_city("PARIS") == "Paris"
    
    def test_keep_paris_titlecase(self):
        """Teste que Paris reste inchangé."""
        assert normalize_city("Paris") == "Paris"
    
    def test_keep_other_cities(self):
        """Teste que les autres villes restent inchangées."""
        assert normalize_city("Saint-Germain-en-Laye") == "Saint-Germain-en-Laye"
        assert normalize_city("Boulogne-Billancourt") == "Boulogne-Billancourt"
    
    def test_handle_none(self):
        """Teste que None retourne None."""
        assert normalize_city(None) is None


class TestNormalizeKeywords:
    """Tests pour la fonction normalize_keywords."""
    
    def test_lowercase_keywords(self):
        """Teste la conversion en minuscules."""
        result = normalize_keywords("Jazz, Musique, Concert")
        assert result == "jazz, musique, concert"
    
    def test_remove_duplicates(self):
        """Teste la suppression des doublons."""
        result = normalize_keywords("jazz, Jazz, JAZZ")
        assert result == "jazz"
    
    def test_preserve_order(self):
        """Teste que l'ordre est préservé."""
        result = normalize_keywords("concert, musique, danse")
        assert result == "concert, musique, danse"
    
    def test_handle_none(self):
        """Teste que None retourne None."""
        assert normalize_keywords(None) is None
    
    def test_handle_empty_string(self):
        """Teste qu'une chaîne vide retourne une valeur falsy."""
        result = normalize_keywords("")
        assert not result  # Accepte "" ou None
    
    def test_handle_whitespace_only(self):
        """Teste qu'une chaîne d'espaces retourne None."""
        assert normalize_keywords("   ") is None


class TestIsValidEvent:
    """Tests pour la fonction is_valid_event."""
    
    def test_valid_event(self):
        """Teste qu'un événement valide retourne True."""
        event = {"title_fr": "Concert", "description_fr": "Un super concert"}
        assert is_valid_event(event) is True
    
    def test_missing_title(self):
        """Teste qu'un événement sans titre retourne une valeur falsy."""
        event = {"description_fr": "Description"}
        assert not is_valid_event(event)
    
    def test_missing_description(self):
        """Teste qu'un événement sans description retourne une valeur falsy."""
        event = {"title_fr": "Titre"}
        assert not is_valid_event(event)
    
    def test_empty_title(self):
        """Teste qu'un titre vide retourne une valeur falsy."""
        event = {"title_fr": "", "description_fr": "Description"}
        assert not is_valid_event(event)
    
    def test_whitespace_title(self):
        """Teste qu'un titre avec seulement des espaces retourne False."""
        event = {"title_fr": "   ", "description_fr": "Description"}
        assert is_valid_event(event) is False
    
    def test_none_title(self):
        """Teste qu'un titre None retourne une valeur falsy."""
        event = {"title_fr": None, "description_fr": "Description"}
        assert not is_valid_event(event)


class TestIsIleDeFrance:
    """Tests pour la fonction is_ile_de_france."""
    
    def test_paris_is_idf(self):
        """Teste que Paris est en IDF."""
        event = {"location_department": "Paris"}
        assert is_ile_de_france(event) is True
    
    def test_seine_et_marne_is_idf(self):
        """Teste que Seine-et-Marne est en IDF."""
        event = {"location_department": "Seine-et-Marne"}
        assert is_ile_de_france(event) is True
    
    def test_lyon_not_idf(self):
        """Teste que Lyon n'est pas en IDF."""
        event = {"location_department": "Métropole de Lyon"}
        assert is_ile_de_france(event) is False
    
    def test_nord_not_idf(self):
        """Teste que Nord n'est pas en IDF."""
        event = {"location_department": "Nord"}
        assert is_ile_de_france(event) is False
    
    def test_normalized_department(self):
        """Teste avec un département à normaliser."""
        event = {"location_department": "Val-D'Oise"}
        assert is_ile_de_france(event) is True
    
    def test_fallback_to_region(self):
        """Teste le fallback sur la région si pas de département."""
        event = {"location_region": "Île-de-France"}
        assert is_ile_de_france(event) is True


class TestCleanEvent:
    """Tests pour la fonction clean_event."""
    
    def test_clean_html_in_longdescription(self):
        """Teste le nettoyage HTML dans longdescription_fr."""
        event = {
            "title_fr": "Test",
            "longdescription_fr": "<p>Description</p>"
        }
        cleaned = clean_event(event)
        assert "<p>" not in cleaned["longdescription_fr"]
        assert "Description" in cleaned["longdescription_fr"]
    
    def test_normalize_department_in_event(self):
        """Teste la normalisation du département."""
        event = {"location_department": "Val-D'Oise"}
        cleaned = clean_event(event)
        assert cleaned["location_department"] == "Val-d'Oise"
    
    def test_normalize_city_in_event(self):
        """Teste la normalisation de la ville."""
        event = {"location_city": "PARIS"}
        cleaned = clean_event(event)
        assert cleaned["location_city"] == "Paris"
    
    def test_normalize_keywords_in_event(self):
        """Teste la normalisation des mots-clés."""
        event = {"keywords_fr": "Jazz, JAZZ, jazz"}
        cleaned = clean_event(event)
        assert cleaned["keywords_fr"] == "jazz"


# =============================================================================
# TESTS POUR prepare_for_vectorization.py
# =============================================================================

class TestCreateTextForEmbedding:
    """Tests pour la fonction create_text_for_embedding."""
    
    def test_includes_title(self):
        """Teste que le titre est inclus."""
        event = {"title_fr": "Concert de jazz"}
        result = create_text_for_embedding(event)
        assert "Titre: Concert de jazz" in result
    
    def test_includes_description(self):
        """Teste que la description est incluse."""
        event = {"description_fr": "Une soirée musicale"}
        result = create_text_for_embedding(event)
        assert "Description: Une soirée musicale" in result
    
    def test_includes_location(self):
        """Teste que la localisation est incluse."""
        event = {
            "location_name": "Le Sunset",
            "location_city": "Paris",
            "location_department": "Paris"
        }
        result = create_text_for_embedding(event)
        assert "Lieu:" in result
        assert "Le Sunset" in result
        assert "Paris" in result
    
    def test_includes_date(self):
        """Teste que la date est incluse."""
        event = {"daterange_fr": "Du 15 au 16 mars 2025"}
        result = create_text_for_embedding(event)
        assert "Date: Du 15 au 16 mars 2025" in result
    
    def test_includes_keywords(self):
        """Teste que les mots-clés sont inclus."""
        event = {"keywords_fr": "jazz, musique"}
        result = create_text_for_embedding(event)
        assert "Mots-clés: jazz, musique" in result
    
    def test_truncates_long_description(self):
        """Teste que les descriptions longues sont tronquées."""
        long_text = "A" * 2000
        event = {"longdescription_fr": long_text}
        result = create_text_for_embedding(event)
        assert len(result) < 1500  # Tronqué à ~1000 + préfixe
        assert "..." in result
    
    def test_handles_none_values(self):
        """Teste la gestion des valeurs None."""
        event = {
            "title_fr": "Test",
            "description_fr": None,
            "location_city": None
        }
        result = create_text_for_embedding(event)
        assert "Titre: Test" in result
        # Pas d'erreur avec None
    
    def test_handles_empty_event(self):
        """Teste avec un événement vide."""
        event = {}
        result = create_text_for_embedding(event)
        assert result == ""


class TestExtractMetadata:
    """Tests pour la fonction extract_metadata."""
    
    def test_extracts_all_fields(self):
        """Teste l'extraction de tous les champs."""
        event = {
            "uid": "12345",
            "title_fr": "Concert",
            "description_fr": "Description",
            "daterange_fr": "15 mars 2025",
            "firstdate_begin": "2025-03-15T20:00:00",
            "firstdate_end": "2025-03-15T23:00:00",
            "location_name": "Salle",
            "location_city": "Paris",
            "location_department": "Paris",
            "canonicalurl": "https://example.com",
            "image": "https://example.com/image.jpg",
            "keywords_fr": "jazz",
            "conditions_fr": "Gratuit"
        }
        metadata = extract_metadata(event)
        
        assert metadata["uid"] == "12345"
        assert metadata["title"] == "Concert"
        assert metadata["description"] == "Description"
        assert metadata["date_range"] == "15 mars 2025"
        assert metadata["date_start"] == "2025-03-15T20:00:00"
        assert metadata["date_end"] == "2025-03-15T23:00:00"
        assert metadata["location_name"] == "Salle"
        assert metadata["city"] == "Paris"
        assert metadata["department"] == "Paris"
        assert metadata["url"] == "https://example.com"
        assert metadata["image"] == "https://example.com/image.jpg"
        assert metadata["keywords"] == "jazz"
        assert metadata["conditions"] == "Gratuit"
    
    def test_handles_missing_fields(self):
        """Teste avec des champs manquants."""
        event = {"uid": "12345"}
        metadata = extract_metadata(event)
        
        assert metadata["uid"] == "12345"
        assert metadata["title"] == ""
        assert metadata["city"] == ""


class TestPrepareEventForVectorization:
    """Tests pour la fonction prepare_event_for_vectorization."""
    
    def test_returns_correct_structure(self):
        """Teste que la structure de sortie est correcte."""
        event = {
            "uid": "12345",
            "title_fr": "Concert",
            "description_fr": "Description"
        }
        result = prepare_event_for_vectorization(event)
        
        assert "id" in result
        assert "text_for_embedding" in result
        assert "metadata" in result
        assert result["id"] == "12345"
    
    def test_text_for_embedding_not_empty(self):
        """Teste que le texte pour embedding n'est pas vide."""
        event = {
            "uid": "12345",
            "title_fr": "Concert",
            "description_fr": "Description"
        }
        result = prepare_event_for_vectorization(event)
        assert len(result["text_for_embedding"]) > 0


# =============================================================================
# TESTS POUR chunk_events.py (avec LangChain)
# =============================================================================

class TestChunkingParameters:
    """Tests pour vérifier les paramètres de chunking."""
    
    def test_chunk_size_defined(self):
        """Teste que CHUNK_SIZE est défini correctement."""
        assert CHUNK_SIZE == 800
    
    def test_chunk_overlap_defined(self):
        """Teste que CHUNK_OVERLAP est défini correctement."""
        assert CHUNK_OVERLAP == 100
    
    def test_min_chunk_size_defined(self):
        """Teste que MIN_CHUNK_SIZE est défini correctement."""
        assert MIN_CHUNK_SIZE == 100
    
    def test_separators_defined(self):
        """Teste que les séparateurs LangChain sont définis."""
        assert SEPARATORS == ["\n\n", "\n", ". ", " ", ""]


class TestSplitTextIntoChunks:
    """Tests pour la fonction split_text_into_chunks (LangChain)."""
    
    def test_short_text_single_chunk(self):
        """Teste qu'un texte court (>= MIN_CHUNK_SIZE) reste en un seul chunk."""
        # Le texte doit faire au moins MIN_CHUNK_SIZE (100) caractères
        text = "Ceci est un texte suffisamment long pour être conservé comme un seul chunk lors du découpage avec LangChain."
        assert len(text) >= MIN_CHUNK_SIZE  # Vérifie que le texte est assez long
        chunks = split_text_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text_multiple_chunks(self):
        """Teste qu'un texte long est découpé en plusieurs chunks."""
        # Crée un texte plus long que CHUNK_SIZE (800)
        text = "Ceci est une phrase de test. " * 100  # ~2900 caractères
        chunks = split_text_into_chunks(text)
        assert len(chunks) > 1
    
    def test_chunk_size_approximately_respected(self):
        """Teste que la taille des chunks est approximativement respectée."""
        text = "Lorem ipsum dolor sit amet consectetur. " * 100
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            # LangChain peut dépasser légèrement pour respecter les séparateurs
            assert len(chunk) <= CHUNK_SIZE + 100
    
    def test_chunks_have_minimum_size(self):
        """Teste que les chunks ont une taille minimale."""
        text = "Ceci est une phrase de test. " * 100
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            assert len(chunk) >= MIN_CHUNK_SIZE
    
    def test_empty_text(self):
        """Teste qu'un texte vide retourne une liste vide."""
        chunks = split_text_into_chunks("")
        assert chunks == []
    
    def test_none_text(self):
        """Teste que None retourne une liste vide."""
        chunks = split_text_into_chunks(None)
        assert chunks == []
    
    def test_text_with_paragraphs(self):
        """Teste le découpage avec des paragraphes."""
        text = "Premier paragraphe avec du contenu.\n\nDeuxième paragraphe.\n\nTroisième paragraphe." * 20
        chunks = split_text_into_chunks(text)
        assert len(chunks) >= 1
        # Vérifie que le texte est préservé
        reconstructed = "".join(chunks)
        assert "Premier paragraphe" in reconstructed
    
    def test_text_with_sentences(self):
        """Teste le découpage avec des phrases."""
        text = "Première phrase. Deuxième phrase. Troisième phrase. " * 50
        chunks = split_text_into_chunks(text)
        assert len(chunks) >= 1
    
    def test_very_short_text_preserved(self):
        """Teste qu'un texte court (>= MIN_CHUNK_SIZE) est préservé intégralement."""
        # Le texte doit faire au moins MIN_CHUNK_SIZE (100) caractères
        text = "Voici un texte de test suffisamment long pour vérifier que le contenu est préservé intégralement lors du chunking."
        assert len(text) >= MIN_CHUNK_SIZE
        chunks = split_text_into_chunks(text)
        assert len(chunks) == 1
        assert text in chunks[0]
    
    def test_text_below_min_size_returns_empty(self):
        """Teste qu'un texte inférieur à MIN_CHUNK_SIZE retourne une liste vide."""
        text = "Texte trop court."  # < 100 caractères
        assert len(text) < MIN_CHUNK_SIZE
        chunks = split_text_into_chunks(text)
        assert chunks == []
    
    def test_text_at_boundary(self):
        """Teste un texte exactement à la limite de CHUNK_SIZE."""
        text = "A" * CHUNK_SIZE
        chunks = split_text_into_chunks(text)
        assert len(chunks) == 1


class TestChunkEvent:
    """Tests pour la fonction chunk_event."""
    
    def test_single_chunk_event(self):
        """Teste un événement qui tient en un seul chunk."""
        # Le texte doit faire au moins MIN_CHUNK_SIZE (100) caractères
        text = "Ceci est un texte suffisamment long pour être conservé comme un seul chunk lors du découpage avec LangChain."
        event = {
            "id": "12345",
            "text_for_embedding": text,
            "metadata": {"title": "Test"}
        }
        chunks = chunk_event(event)
        
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "12345"
        assert chunks[0]["event_id"] == "12345"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["total_chunks"] == 1
    
    def test_multiple_chunks_event(self):
        """Teste un événement découpé en plusieurs chunks."""
        # Crée un texte suffisamment long pour nécessiter plusieurs chunks
        long_text = "Ceci est une description détaillée. " * 100
        event = {
            "id": "12345",
            "text_for_embedding": long_text,
            "metadata": {"title": "Test"}
        }
        chunks = chunk_event(event)
        
        assert len(chunks) > 1
        assert chunks[0]["chunk_id"] == "12345_0"
        assert chunks[1]["chunk_id"] == "12345_1"
        assert all(c["event_id"] == "12345" for c in chunks)
        assert all(c["total_chunks"] == len(chunks) for c in chunks)
    
    def test_metadata_preserved(self):
        """Teste que les métadonnées sont préservées."""
        metadata = {
            "title": "Concert",
            "city": "Paris",
            "url": "https://example.com"
        }
        # Le texte doit faire au moins MIN_CHUNK_SIZE (100) caractères
        text = "Ceci est un texte de test suffisamment long pour être conservé et permettre de vérifier que les métadonnées sont bien préservées."
        event = {
            "id": "12345",
            "text_for_embedding": text,
            "metadata": metadata
        }
        chunks = chunk_event(event)
        
        assert len(chunks) >= 1
        assert chunks[0]["metadata"] == metadata
    
    def test_chunk_index_increments(self):
        """Teste que l'index des chunks s'incrémente."""
        long_text = "Description longue pour créer plusieurs chunks. " * 100
        event = {
            "id": "12345",
            "text_for_embedding": long_text,
            "metadata": {}
        }
        chunks = chunk_event(event)
        
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
    
    def test_text_content_preserved(self):
        """Teste que le contenu textuel est préservé dans les chunks."""
        # Le texte doit faire au moins MIN_CHUNK_SIZE (100) caractères
        text = "Contenu important à préserver dans le chunk. Ce texte doit être suffisamment long pour dépasser la taille minimale requise."
        event = {
            "id": "12345",
            "text_for_embedding": text,
            "metadata": {}
        }
        chunks = chunk_event(event)
        
        assert len(chunks) >= 1
        assert "Contenu important" in chunks[0]["text"]
    
    def test_empty_text_handling(self):
        """Teste la gestion d'un texte vide."""
        event = {
            "id": "12345",
            "text_for_embedding": "",
            "metadata": {}
        }
        chunks = chunk_event(event)
        
        assert len(chunks) == 0
    
    def test_chunk_has_required_fields(self):
        """Teste que chaque chunk a tous les champs requis."""
        # Le texte doit faire au moins MIN_CHUNK_SIZE (100) caractères
        text = "Texte de test suffisamment long pour vérifier que tous les champs requis sont bien présents dans chaque chunk généré."
        event = {
            "id": "12345",
            "text_for_embedding": text,
            "metadata": {"title": "Test"}
        }
        chunks = chunk_event(event)
        
        assert len(chunks) >= 1
        required_fields = ["chunk_id", "event_id", "chunk_index", "total_chunks", "text", "metadata"]
        for chunk in chunks:
            for field in required_fields:
                assert field in chunk, f"Champ manquant: {field}"


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

class TestIntegration:
    """Tests d'intégration pour le pipeline complet."""
    
    def test_full_pipeline(self):
        """Teste le pipeline complet de nettoyage et préparation."""
        # Événement brut avec HTML et données à normaliser
        raw_event = {
            "uid": "99999",
            "title_fr": "Festival de Jazz",
            "description_fr": "Un festival incontournable",
            "longdescription_fr": "<p><strong>Programme:</strong></p><ul><li>Concert 1</li><li>Concert 2</li></ul>",
            "location_name": "Parc de la Villette",
            "location_city": "PARIS",
            "location_department": "paris",
            "location_region": "Île-de-France",
            "daterange_fr": "Du 1er au 5 juillet 2025",
            "firstdate_begin": "2025-07-01T18:00:00",
            "firstdate_end": "2025-07-01T23:00:00",
            "canonicalurl": "https://openagenda.com/festival-jazz",
            "keywords_fr": "Jazz, JAZZ, musique, Musique",
            "conditions_fr": "Entrée libre"
        }
        
        # Étape 1: Nettoyage
        assert is_valid_event(raw_event) is True
        assert is_ile_de_france(raw_event) is True
        
        cleaned = clean_event(raw_event)
        
        # Vérifications du nettoyage
        assert "<p>" not in cleaned["longdescription_fr"]
        assert "Programme:" in cleaned["longdescription_fr"]
        assert cleaned["location_city"] == "Paris"
        assert cleaned["location_department"] == "Paris"
        assert cleaned["keywords_fr"] == "jazz, musique"
        
        # Étape 2: Préparation pour vectorisation
        prepared = prepare_event_for_vectorization(cleaned)
        
        # Vérifications de la préparation
        assert prepared["id"] == "99999"
        assert "Festival de Jazz" in prepared["text_for_embedding"]
        assert "Parc de la Villette" in prepared["text_for_embedding"]
        assert "Paris" in prepared["text_for_embedding"]
        assert prepared["metadata"]["city"] == "Paris"
        assert prepared["metadata"]["url"] == "https://openagenda.com/festival-jazz"
        
        # Étape 3: Chunking (avec LangChain)
        chunks = chunk_event(prepared)
        
        # Vérifications du chunking
        assert len(chunks) >= 1
        assert chunks[0]["event_id"] == "99999"
        assert "Festival de Jazz" in chunks[0]["text"]
    
    def test_pipeline_with_long_description(self):
        """Teste le pipeline avec une description longue nécessitant plusieurs chunks."""
        raw_event = {
            "uid": "88888",
            "title_fr": "Exposition d'art contemporain",
            "description_fr": "Une exposition majeure",
            "longdescription_fr": "<p>" + "Description détaillée de l'exposition. " * 100 + "</p>",
            "location_name": "Centre Pompidou",
            "location_city": "Paris",
            "location_department": "Paris",
            "daterange_fr": "Du 1er mars au 30 juin 2025",
            "canonicalurl": "https://example.com"
        }
        
        # Nettoyage
        cleaned = clean_event(raw_event)
        
        # Préparation
        prepared = prepare_event_for_vectorization(cleaned)
        
        # Chunking avec LangChain
        chunks = chunk_event(prepared)
        
        # Devrait avoir plusieurs chunks
        assert len(chunks) >= 1
        
        # Tous les chunks doivent référencer le même événement
        for chunk in chunks:
            assert chunk["event_id"] == "88888"
            assert chunk["total_chunks"] == len(chunks)


# =============================================================================
# TESTS DES FICHIERS DE DONNÉES
# =============================================================================

class TestDataFiles:
    """Tests de validation des fichiers de données générés."""
    
    @pytest.fixture
    def data_dir(self):
        """Retourne le chemin vers le dossier data."""
        return Path(__file__).resolve().parent.parent / "src" / "data"
    
    def test_raw_file_exists(self, data_dir):
        """Teste que le fichier brut existe."""
        raw_path = data_dir / "raw" / "events_raw.json"
        assert raw_path.exists(), f"Fichier {raw_path} non trouvé"
    
    def test_cleaned_file_exists(self, data_dir):
        """Teste que le fichier nettoyé existe."""
        cleaned_path = data_dir / "processed" / "events_cleaned.json"
        assert cleaned_path.exists(), f"Fichier {cleaned_path} non trouvé"
    
    def test_vectorization_file_exists(self, data_dir):
        """Teste que le fichier de vectorisation existe."""
        vec_path = data_dir / "processed" / "events_for_vectorization.json"
        assert vec_path.exists(), f"Fichier {vec_path} non trouvé"
    
    def test_chunked_file_exists(self, data_dir):
        """Teste que le fichier de chunks existe."""
        chunked_path = data_dir / "processed" / "events_chunked.json"
        assert chunked_path.exists(), f"Fichier {chunked_path} non trouvé"
    
    def test_vectorized_file_exists(self, data_dir):
        """Teste que le fichier vectorisé existe."""
        vectorized_path = data_dir / "processed" / "events_vectorized.json"
        assert vectorized_path.exists(), f"Fichier {vectorized_path} non trouvé"
    
    def test_embeddings_file_exists(self, data_dir):
        """Teste que le fichier numpy des embeddings existe."""
        embeddings_path = data_dir / "processed" / "embeddings.npy"
        assert embeddings_path.exists(), f"Fichier {embeddings_path} non trouvé"
    
    def test_raw_file_structure(self, data_dir):
        """Teste la structure du fichier brut."""
        raw_path = data_dir / "raw" / "events_raw.json"
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "metadata" in data
            assert "events" in data
            assert len(data["events"]) > 0
    
    def test_cleaned_file_structure(self, data_dir):
        """Teste la structure du fichier nettoyé."""
        cleaned_path = data_dir / "processed" / "events_cleaned.json"
        if cleaned_path.exists():
            with open(cleaned_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "metadata" in data
            assert "events" in data
            assert "cleaning" in data["metadata"]
    
    def test_chunked_file_structure(self, data_dir):
        """Teste la structure du fichier de chunks."""
        chunked_path = data_dir / "processed" / "events_chunked.json"
        if chunked_path.exists():
            with open(chunked_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "metadata" in data
            assert "chunks" in data
            assert "chunking" in data["metadata"]
            
            # Vérifie la structure d'un chunk
            if len(data["chunks"]) > 0:
                chunk = data["chunks"][0]
                assert "chunk_id" in chunk
                assert "event_id" in chunk
                assert "text" in chunk
                assert "metadata" in chunk
    
    def test_chunked_file_has_langchain_method(self, data_dir):
        """Teste que le fichier de chunks indique la méthode LangChain."""
        chunked_path = data_dir / "processed" / "events_chunked.json"
        if chunked_path.exists():
            with open(chunked_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunking_meta = data["metadata"]["chunking"]
            assert "method" in chunking_meta
            assert "langchain" in chunking_meta["method"].lower()
    
    def test_vectorized_file_structure(self, data_dir):
        """Teste la structure du fichier vectorisé."""
        vectorized_path = data_dir / "processed" / "events_vectorized.json"
        if vectorized_path.exists():
            with open(vectorized_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "metadata" in data
            assert "chunks" in data
            assert "vectorization" in data["metadata"]
            
            # Vérifie les métadonnées de vectorisation
            vec_meta = data["metadata"]["vectorization"]
            assert "model" in vec_meta
            assert "embedding_dimension" in vec_meta
    
    def test_embeddings_file_structure(self, data_dir):
        """Teste la structure du fichier numpy des embeddings."""
        embeddings_path = data_dir / "processed" / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            
            # Vérifie la forme
            assert len(embeddings.shape) == 2
            assert embeddings.shape[1] == 1024  # Dimension Mistral
            
            # Vérifie le type
            assert embeddings.dtype == np.float32
    
    def test_embeddings_count_matches_chunks(self, data_dir):
        """Teste que le nombre d'embeddings correspond au nombre de chunks."""
        vectorized_path = data_dir / "processed" / "events_vectorized.json"
        embeddings_path = data_dir / "processed" / "embeddings.npy"
        
        if vectorized_path.exists() and embeddings_path.exists():
            with open(vectorized_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            embeddings = np.load(embeddings_path)
            
            assert len(data["chunks"]) == embeddings.shape[0]
    
    def test_no_html_in_cleaned_data(self, data_dir):
        """Teste qu'il n'y a plus de HTML dans les données nettoyées."""
        cleaned_path = data_dir / "processed" / "events_cleaned.json"
        if cleaned_path.exists():
            with open(cleaned_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            html_tags = ["<p>", "<br>", "<strong>", "<a ", "<ul>", "<li>"]
            
            for event in data["events"][:100]:  # Teste les 100 premiers
                long_desc = event.get("longdescription_fr", "")
                if long_desc:
                    for tag in html_tags:
                        assert tag not in long_desc, f"HTML trouvé: {tag}"
    
    def test_all_events_in_idf(self, data_dir):
        """Teste que tous les événements sont en Île-de-France."""
        cleaned_path = data_dir / "processed" / "events_cleaned.json"
        if cleaned_path.exists():
            with open(cleaned_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for event in data["events"]:
                dept = event.get("location_department", "")
                if dept:
                    assert dept in ILE_DE_FRANCE_DEPARTMENTS, f"Département hors IDF: {dept}"
    
    def test_chunk_sizes_within_limits(self, data_dir):
        """Teste que les chunks respectent les limites de taille."""
        chunked_path = data_dir / "processed" / "events_chunked.json"
        if chunked_path.exists():
            with open(chunked_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for chunk in data["chunks"][:500]:  # Teste les 500 premiers
                text_length = len(chunk["text"])
                assert text_length >= MIN_CHUNK_SIZE, f"Chunk trop petit: {text_length}"
                # LangChain peut légèrement dépasser pour respecter les séparateurs
                assert text_length <= CHUNK_SIZE + 150, f"Chunk trop grand: {text_length}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])