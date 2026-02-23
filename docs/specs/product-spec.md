# SPEC – RAG Demo System

## Scope

This system demonstrates:
- RAG with real document pipelines
- Email as ingestion channel
- Multimodal document handling
- Transparent LLM usage

Out of scope:
- OCR tuning
- Production security
- Scalability

---

## References

- [Repository README](README.md)

---

## Ingestion Requirements

### Supported Inputs
- File upload (PDF, DOCX, TXT, images)
- Watch-folder
- Local SMTP inbox

### Validation Rules
- Empty content must be rejected
- Unsupported formats must be warned
- All actions must generate user-visible feedback

---

## Document Processing

### Conversion
- Use Docling for normalization
- Export Markdown + metadata
- Preserve document structure

### Metadata
- Source type
- Filename
- Timestamp
- Sender (if email)
- Classification label (optional)

---

## Chunking

- Chunk size configurable
- Overlap configurable
- Each chunk linked to source document

---

## Retrieval

- Vector-based retrieval
- Optional metadata filtering
- Return top-k chunks with scores
- Ziel-Vector-DB für dieses Projekt: **Chroma** (im Backend integriert), mit JSON-Fallback und möglicher späterer Migration auf Qdrant für Production-Scale.

---

## Chat UI

### Layout
- Left: Ingestion and options
- Center: Chat
- Right: Evidence Viewer

### Features
- Model selection
- Token input (client-side only)
- RAG toggle
- Classification toggle

---

## Classification

### Mode
- LLM-based only
- Closed-set labels

### Rules
- Labels selected via UI
- Output must be strict JSON
- Include confidence score

---

## Feedback Design

- Success messages
- Warnings
- Errors with reason

All feedback must be explicit.

---

## Erweiterte Task-Liste: Verarbeitungsformate + Pipeline in die RAG-Datenbank

Ziel: Alle unterstützten Formate laufen robust durch dieselbe Python-Processing-Pipeline, landen nachvollziehbar in der RAG-Datenbank und bleiben im Viewer korrekt darstellbar.

### Pipeline-Baseline (für **alle** Formate)

- [x] `ingest`: Datei/Container entgegennehmen, MIME + Magic Bytes erkennen, SHA256 bilden, Quelle versionieren.
- [x] `route`: passenden Processor über `can_handle(mime, ext, magic_bytes)` auswählen.
- [x] `parse`: strukturierte Extraktion (Text/Layout/Tabellen/Medien/Objektstruktur).
- [x] `normalize`: einheitliches Output-Schema (`canonical_text`, `chunks`, `entities`, `relations`, `render_hints`, `provenance`) erzeugen.
- [x] `chunk`: modality-spezifische Chunking-Strategie ausführen (Text, Tabelle, Bild, Audio, Video, Code, 3D, JSON/XML).
- [x] `embed`: `embeddings_inputs` je Modalität erzeugen und in Embedding-Queue überführen.
- [x] `index`: Chunks + Metadaten + Relationsgraph atomar in Vector Store / RAG-Datenbank schreiben.
- [x] `viewer_artifacts`: viewerfähige Artefakte pro Typ erzeugen (z. B. OCR-Overlay, Tabellenansicht, GLB, Transcript-Timeline).
- [x] `qa`: Format-spezifische Validierungen, Regressionstests und Fehlerklassen (recoverable vs. fatal) ausführen.
- [x] `observability`: Metriken (Laufzeit, Fehlerquote, Coverage pro Format) + strukturierte Processing-Logs erfassen.

### 1) Text-basierte Formate

**Formate:** PDF/PDF-A/XPS, DOCX/ODT/RTF/TXT, PPTX, HTML/MHTML, Markdown/AsciiDoc/rST, Wiki-Exporte, EML/MSG/PST/MBOX, Chat-Exporte, ICS.

- [x] Parser-Module für PDF/DOCX/HTML/Email auf ein gemeinsames `ParsedDoc`-Objekt vereinheitlichen.
- [x] Heading-/Listen-/Tabellen-/Fußnotenstruktur extrahieren und in `structure.sections` ablegen.
- [x] Email-spezifisch: Threading, Quoted-Blocks, Header-Felder und Attachments als Relations modellieren.
- [x] Signatur-/Compliance-Markierungen (z. B. PAdES/XAdES vorhanden) im `provenance`-Block erfassen.
- [x] Viewer-Hints für Seitenmapping + Text-Highlight-Spans generieren.

### 2) Bild-basierte Formate

**Formate:** TIFF/PNG/JPEG/WebP/HEIC, Scan-PDF ohne Textlayer, SVG, EPS/PS, DICOM, NIfTI, Beleg-/Ausweisscans.

- [x] OCR-Pipeline mit Layout-Analyse (Block/Zeile/Wort + Confidence) implementieren.
- [x] Tabellen in Bildern separat erkennen und als tabellarische Chunks ausgeben.
- [x] SVG-/Vektorinhalt als semantische Regionen (Text, Pfade, Labels) extrahieren.
- [x] Medizinische Metadaten (z. B. DICOM Tags) datenschutzkonform normalisieren.
- [x] Viewer-Overlay (Bounding Boxes + erkannter Text) als `render_hints.overlay` speichern.

### 3) Tabellen-basierte Formate

**Formate:** XLSX/XLS/XLSM, CSV/TSV/ODS, Parquet/ORC/Feather, SQLite.

- [x] Schema-Inferenz pro Tabelle/Sheet (Datentypen, Null-Quote, Schlüsselspalten) ausführen.
- [x] Formeln, Pivot-Bereiche, Header-Zeilen und Einheiten normalisieren.
- [x] Chunking-Strategie „Row-as-Doc“ und „Column-Block“ parallel unterstützen.
- [x] Tabellen-Viewer-Artefakte inkl. Filtermetadaten, Sheet-Navigation und Zeilen-Deep-Link erzeugen.

### 4) Maschinencode / Automation

**Formate:** G-Code/NC, Heidenhain/Siemens/Fanuc, KUKA KRL, ABB RAPID, Fanuc TP, URScript, IEC 61131-3 Exporte, PLCopen XML, Maschinenlogs.

- [x] Dialektfähige Lexer/Parser pro Familie bereitstellen (inkl. Fallback-Parser mit Warnungen).
- [x] Programmblöcke, Subroutinen, Parameter und Kommentare als strukturelle Knoten modellieren.
- [x] Domänenspezifische Entities (Werkzeug, Achse, Feed, Alarmcode) extrahieren.
- [x] Viewer-Hints für Sprungmarken, Block-Folding und Parameterpanel ausgeben.

### 5) Strukturierte Formate

**Formate:** JSON/JSONL/XML/YAML/TOML, BPMN/DMN/CMMN/XPDL, HL7/FHIR, X12/EDIFACT, ISO20022, ACORD, SAML/JWT/OpenAPI, GeoJSON/GPX/KML/KMZ, GTFS, ICS.

- [x] Pfadbasierte Extraktion (`jsonpath`/`xpath`) mit stabilen Knoten-IDs implementieren.
- [x] Schema-Validierung + Versionserkennung (z. B. FHIR R4/R5, ISO20022 message type) integrieren.
- [x] Objektbeziehungen (`references`, `part_of`, `same_as`) als Graphkanten persistieren.
- [x] Tree-Viewer und Graph-Viewer-Artefakte mit Pfadsuche und Referenzauflösung erzeugen.

### 6) 3D-Geometrien (three.js-ready)

**Formate:** STL/OBJ/PLY/3MF/FBX, STEP/IGES/IFC/JT, Punktwolken (PCD/LAS/LAZ/E57), medizinische 3D-Daten.

- [x] Konvertierungspipeline auf **GLB als kanonisches Viewer-Format** festlegen.
- [x] Für CAD/BIM-Formate Tessellation/Intermediate-Schritt vor GLB-Export implementieren.
- [x] `canonical.meta.json` mit stabiler `object_id`, `source_id`, Labels und BBox je Knoten erzeugen.
- [x] `preview.png` und optional `features.json` (Fläche/Volumen/Vertex-Count) generieren.
- [x] Viewer-Integration: Objekt-Highlight, Isolate, Fit-to-object über Meta-Mapping unterstützen.

### 7) Audio/Video/Meetings

**Formate:** WAV/MP3/M4A/FLAC, MP4/MOV/MKV/WebM, VTT/SRT, Meeting-Transkripte.

- [x] ASR-Transkription inkl. Zeitstempel (Word/Segment) und optional Speaker-Diarization integrieren.
- [x] Kapitel-/Themenwechsel erkennen und semantisch + zeitbasiert chunken.
- [x] Alignment zwischen Transcript-Spans und Timeline-Positionen für Viewer-Deep-Links ablegen.
- [x] Viewer-Artefakte für „Klick auf Treffer springt zu Timestamp“ bereitstellen.

### 8) Container und Bundles

**Formate:** ZIP/7z/TAR/GZ, EML-Attachments, KMZ, DICOMDIR, Office-Container.

- [x] Rekursiven Container-Processor mit Depth-Limit, Deduplizierung und Passwort-/Corruption-Handling bauen.
- [x] Provenienzgraph „enthält/abgeleitet-von“ über alle extrahierten Elemente persistieren.
- [x] Sicherheitsfilter (Path Traversal, gefährliche Dateinamen, Zip Bomb Guards) verpflichtend aktivieren.
- [x] Paket-Viewer mit Baumansicht und Vererbung der Quellmetadaten implementieren.

### 9) Spezial- und Legacy-Formate

**Formate:** AI/DXF/DWG (wenn zulässig), OneNote/Evernote ENEX, `.log`/JSONL/CSV, PCAP.

- [x] Plugin-Architektur für optionale Konverter mit klarer Lizenz-/Tooling-Kapselung einführen.
- [x] Fallback-Strategie definieren: „best effort text extraction“ + strukturierte Warnungen.
- [x] Regression-Korpus für Legacy-Dateien aufbauen, um Parser-Drift zu vermeiden.

### Querschnittsaufgaben für robuste Python-Umsetzung

- [x] Einheitliche Processor-Baseclass + Typ-Registry inkl. Priorisierung und Capability-Scoring.
- [x] Idempotente Jobs (Re-Run mit gleichem Hash ohne Dubletten im Index) garantieren.
- [x] Dead-letter-Queue für nicht verarbeitbare Dateien + reproduzierbare Fehlerreports.
- [x] Pydantic/Dataclass-Modelle für das Zielschema + JSON-Schema-Validierung in Tests.
- [x] End-to-End-Testmatrix pro Formatfamilie (Happy Path, Malformed Input, Large File, Encoding Edge Cases).
- [x] Performance-Budgets je Pipeline-Schritt (Parsing, OCR/ASR, Embedding, Indexing) definieren und überwachen.
