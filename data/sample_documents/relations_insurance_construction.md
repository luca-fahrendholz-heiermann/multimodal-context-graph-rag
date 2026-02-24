# Multimodal Context Graph Relations

## Insurance

### 01_anfrage_email.eml → 02_schadensmeldung.pdf

**type:** references\
**note:**\
Die E-Mail dokumentiert die initiale Schadensanzeige und verweist
explizit auf die formale Schadensmeldung, in der der Schaden
strukturiert beschrieben, kategorisiert und mit einer Claim-ID versehen
wird.

### 02_schadensmeldung.pdf → 04_kostenvoranschlag.xlsx

**type:** references\
**note:**\
Die Schadensmeldung beschreibt einzelne Schadenspositionen, die im
Kostenvoranschlag rechnerisch bewertet und als tabellarische Positionen
mit Mengen und Einzelpreisen quantifiziert werden.

### 04_kostenvoranschlag.xlsx → 02_schadensmeldung.pdf

**type:** derived_from\
**note:**\
Der Kostenvoranschlag wurde direkt aus den in der Schadensmeldung
beschriebenen Positionen abgeleitet und stellt die finanzielle
Berechnung der gemeldeten Schäden dar.

### 02_schadensmeldung.pdf → 03_policenvertrag.pdf

**type:** references\
**note:**\
Die Schadensmeldung bezieht sich auf den zugrunde liegenden
Policenvertrag, um die Deckungsfähigkeit des gemeldeten Schadens anhand
der vertraglich vereinbarten Klauseln zu prüfen.

### 03_policenvertrag.pdf → 05_versicherungsschein.xml

**type:** related_to\
**note:**\
Der Policenvertrag in PDF-Form und der strukturierte Versicherungsschein
im XML-Format repräsentieren denselben Versicherungsvertrag in
unterschiedlichen Darstellungsformen, wobei das PDF die juristische
Lesefassung und das XML die maschinenlesbare Struktur abbildet.

### 05_versicherungsschein.xml → 03_policenvertrag.pdf

**type:** references\
**note:**\
Der strukturierte Versicherungsschein enthält die kanonische Policen-ID
und verweist damit logisch auf den vollständigen Vertragsinhalt im
zugehörigen Policenvertrag als juristische Referenz.

### 01_anfrage_email.eml → 03_policenvertrag.pdf

**type:** references\
**note:**\
Die E-Mail nennt die Policen-ID und stellt damit den direkten Bezug
zwischen der Schadensanfrage und dem zugrunde liegenden
Versicherungsvertrag her.

### 05_versicherungsschein.xml → 02_schadensmeldung.pdf

**type:** related_to\
**note:**\
Der Versicherungsschein enthält strukturierte Referenzen auf gemeldete
Schadenfälle und stellt dadurch die formale Zuordnung des Claims zum
entsprechenden Versicherungsvertrag her.

------------------------------------------------------------------------

## Construction

### 01_bauanfrage.eml → 03_bausubmission.pdf

**type:** references\
**note:**\
Die Bauanfrage-E-Mail verweist auf die formale Bausubmission als
strukturiertes Dokument, das Leistungsbeschreibung, Anforderungen und
Vergabebedingungen für das beantragte Projekt konkretisiert.

### 01_bauanfrage.eml → 02_plan_grundriss.dxf

**type:** references\
**note:**\
Die Bauanfrage nimmt Bezug auf den beigefügten Grundrissplan, der die
räumliche und geometrische Grundlage des beantragten Bauvorhabens
definiert.

### 01_bauanfrage.eml → 04_kostenschaetzung.xlsx

**type:** references\
**note:**\
Die E-Mail benennt die Kostenschätzung als wirtschaftliche Bewertung des
Projekts und stellt damit die Verbindung zwischen formaler Anfrage und
finanzieller Planung her.

### 03_bausubmission.pdf → 04_kostenschaetzung.xlsx

**type:** references\
**note:**\
Die Kostenschätzung basiert auf den im Leistungsverzeichnis der
Submission beschriebenen Positionen und quantifiziert diese in monetärer
Form.

### 04_kostenschaetzung.xlsx → 03_bausubmission.pdf

**type:** derived_from\
**note:**\
Die tabellarische Kostenschätzung wurde aus den Leistungspositionen der
Submission abgeleitet und bildet deren wirtschaftliche Auswertung ab.

### 02_plan_grundriss.dxf → building_model.ifc

**type:** related_to\
**note:**\
Der DXF-Grundriss und das IFC-Modell repräsentieren dasselbe Bauprojekt
in unterschiedlichen Modellierungsstufen, wobei der DXF-Plan die
2D-Geometrie und das IFC-Modell die semantisch angereicherte
BIM-Struktur enthält.

### building_model.ifc → building_model.obj

**type:** derived_from\
**note:**\
Das OBJ-Modell wurde aus dem IFC-BIM-Modell exportiert und stellt eine
triangulierte Visualisierungsrepräsentation der semantischen
Gebäudestruktur dar.

### building_model.obj → building_element.obj

**type:** contains\
**note:**\
Das Gesamtmodell im OBJ-Format enthält einzelne Bauelemente, von denen
das separat gespeicherte building_element.obj eine extrahierte
Teilkomponente darstellt.

### building_element.obj → building_element_scan.ply

**type:** related_to\
**note:**\
Das triangulierte Bauelement-Modell steht in Beziehung zur gescannten
Punktwolke desselben Elements, wobei die PLY-Datei eine gemessene
Ist-Geometrie und das OBJ-Modell eine konstruktive Soll-Geometrie
repräsentiert.

### building_model.ifc → 04_kostenschaetzung.xlsx

**type:** related_to\
**note:**\
Das IFC-Modell enthält Bauteil- und Mengeninformationen, die für die
Ermittlung der in der Kostenschätzung aufgeführten Positionen
herangezogen werden können.
