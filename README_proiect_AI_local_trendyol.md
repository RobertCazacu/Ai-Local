# Proiect: AI local pentru mapare categorii (Trendyol) + interfață (Streamlit)

Acest proiect rulează local pe Windows și te ajută să:
- înveți din produsele deja listate (care au categorie asignată)
- propui automat categoria pentru produse noi din site
- opțional: convertești categoria text în category_id Trendyol folosind folderul de fișiere “Trendyol Categories” (ID-ul este în numele fișierului)

Proiectul folosește:
- Ollama (local) pentru embeddings (vectori) din text (Nume/Brand/Descriere)
- un motor de similaritate (în varianta actuală: cosine similarity cu NumPy; suficient pentru ~7.000 produse)
- Streamlit pentru UI (upload fișier -> build/predict -> download rezultat)
- rapidfuzz pentru potrivirea “categorie text -> ID” (fuzzy match) din numele fișierelor de categorii

------------------------------------------------------------
1) Ce face “AI-ul” (explicat simplu)
------------------------------------------------------------

AI-ul nu “inventează” categoria, ci se bazează pe produse similare pe care le ai deja mapate corect.

Flux:
1) Pentru fiecare produs deja mapat (training), construiește un text din:
   - Nume
   - Brand
   - Descriere
2) Transformă textul în embedding (vector) folosind Ollama (model: nomic-embed-text).
3) Salvează embeddings și meta (produse + categoria lor).
4) Pentru un produs nou:
   - calculează embedding
   - caută cei mai similari K vecini în training (K=15 de obicei)
   - face “vot ponderat” pe categorii: categoria cu scorul total cel mai mare devine propunerea
   - calculează scor de încredere (Scor_incredere) și marchează Necesita_verificare dacă e sub prag

Rezultatul este un Excel cu:
- Categorie_propusa (ideal CategoryID)
- Scor_incredere
- Necesita_verificare
- Top_matchuri

------------------------------------------------------------
2) Cerințe (Windows)
------------------------------------------------------------

- Windows x64
- Python 3.11 (recomandat)
- Ollama instalat și pornit
- (opțional) Microsoft C++ Build Tools (a fost necesar în procesul nostru pentru anumite librării native; proiectul actual rulează și fără, dar nu strică să fie instalate)
- acces la fișierele tale Excel:
  - labeled_trendyol.xlsx (training)
  - new.xlsx (produse noi)
  - folder “Trendyol Categories” (XLSX-uri cu ID + nume în denumire)

------------------------------------------------------------
3) Setup inițial (o singură dată)
------------------------------------------------------------

3.1 Creează proiectul
Exemplu:
C:\Users\manue\Desktop\Proiect Ollama\

În acest folder vei avea fișierele:
- categorize_engine.py
- app.py
- build_trendyol_taxonomy.py
- (Excel) labeled_trendyol.xlsx / labeled_trendyol_id.xlsx / new.xlsx
- (folder) Trendyol Categories\

3.2 Creează și activează venv (PowerShell)
În folderul proiectului:

py -3.11 -m venv .venv
.\.venv\Scripts\activate

3.3 Instalează dependințe
python -m pip install -U pip setuptools wheel
pip install pandas openpyxl numpy requests streamlit rapidfuzz

3.4 Instalează / pornește Ollama + modelul de embeddings
Verifică:
ollama --version

Trage model:
ollama pull nomic-embed-text

Ollama rulează local pe:
http://localhost:11434

------------------------------------------------------------
4) Fișierele de input (format)
------------------------------------------------------------

4.1 labeled_trendyol.xlsx (training; categorie text)
Acesta este exportul tău (ex: din easySales/DB) cu produse care au deja categoria asignată pe Trendyol.

Coloane recomandate:
- Nume
- Brand
- Descriere
- Categorie  (text)

Poți avea și:
- SKU / ID produs / link imagine etc. (nu încurcă)

4.2 Folder “Trendyol Categories” (taxonomie + atribute)
Ai mii de fișiere XLSX.
Format denumire:
<ID> <NumeCategorie>.xlsx
ex: 371 Cufflinks.xlsx

În interiorul fiecărui XLSX sunt caracteristicile și valorile acceptate (le folosim după ce avem CategoryID).

4.3 new.xlsx (produse noi)
Exportul din site cu produse pe care vrei să le trimiți în marketplace.

Coloane minime:
- Nume
- Brand (recomandat)
- Descriere (recomandat)

------------------------------------------------------------
5) Pasul A: convertește categoria text în CategoryID (recomandat)
------------------------------------------------------------

De ce:
- în DB ai doar categoria text
- pentru import în marketplace, ID-ul categoriei este mai stabil și mai corect decât textul

Ce face scriptul:
- citește numele fișierelor din folderul “Trendyol Categories”
- extrage category_id și category_name
- generează un catalog (ID + nume + path fișier)
- face un mapping între Categorie (text) din labeled_trendyol.xlsx și category_id
- creează labeled_trendyol_id.xlsx care are o coloană nouă CategoryID

Comandă:

python build_trendyol_taxonomy.py --cat_dir "Trendyol Categories" --labeled "labeled_trendyol.xlsx" --category_col "Categorie" --out_catalog "trendyol_categories_catalog.xlsx" --out_mapping "trendyol_text_to_id_mapping.xlsx" --out_labeled_id "labeled_trendyol_id.xlsx" --min_fuzzy_score 90

Output:
- trendyol_categories_catalog.xlsx
- trendyol_text_to_id_mapping.xlsx
- labeled_trendyol_id.xlsx

Important:
- deschide trendyol_text_to_id_mapping.xlsx și verifică rândurile cu match_type:
  - ambiguous_exact
  - fuzzy_low
  - none
Acestea sunt cazurile care necesită ajustare manuală (o singură dată). După ce le corectezi, poți rerula scriptul sau poți păstra mapping-ul final.

------------------------------------------------------------
6) Pasul B: Build (învățare / indexare)
------------------------------------------------------------

Build creează “memoria” sistemului (embeddings + meta) în folderul modelului.

Recomandat: să faci build din labeled_trendyol_id.xlsx și să folosești label_col = CategoryID.

CLI (PowerShell cu venv activ):

python categorize_engine.py build --labeled labeled_trendyol_id.xlsx --out_dir model_trendyol --text_cols "Nume,Brand,Descriere" --label_col "CategoryID" --workers 4

Dacă vrei să începi mai rapid:
- folosește text_cols = "Nume,Brand" (fără Descriere)

Note de performanță:
- build poate dura mult la 7.000+ produse deoarece se fac embeddings pentru fiecare rând
- descrierile foarte lungi încetinesc
- dacă apar blocaje, scade workers la 2

Ce se creează în out_dir (ex: model_trendyol\):
- embeddings.npy
- meta.csv
- config.json

------------------------------------------------------------
7) Pasul C: Predict (mapare produse noi)
------------------------------------------------------------

Predict primește new.xlsx și scoate rezultat_mapare.xlsx.

CLI:

python categorize_engine.py predict --input new.xlsx --out_dir model_trendyol --output rezultat_mapare.xlsx --k 15 --min_conf 0.55 --workers 4

Interpretare:
- Categorie_propusa: categoria (ID sau text, în funcție de label_col folosit la build)
- Scor_incredere: 0..1
- Necesita_verificare: True dacă e sub prag
- Top_matchuri: motive / vecini cei mai apropiați

Recomandare:
- la început setează min_conf mai sus (0.65–0.75) ca să eviți auto-mapări greșite
- după ce validezi calitatea, coboară spre 0.55

------------------------------------------------------------
8) Interfață (Streamlit) + progress bar
------------------------------------------------------------

Pornește UI:

python -m streamlit run app.py

Se deschide în browser (de obicei):
http://localhost:8501

În UI ai două taburi:
1) Build (învățare)
   - încarci labeled_trendyol_id.xlsx
   - alegi label_col (CategoryID)
   - alegi text_cols (Nume/Brand/Descriere)
   - apeși Build
   - vezi progress bar (X/Y embeddings)

2) Predict (mapare)
   - încarci new.xlsx
   - apeși Predict
   - vezi progress bar
   - descarci rezultat_mapare.xlsx

Important:
- progress bar-ul funcționează pentru embeddings (acolo e timpul mare)
- Streamlit reîncarcă scriptul la fiecare interacțiune; de aceea blocurile trebuie indentate corect

------------------------------------------------------------
9) Cum adaugi produse noi pe viitor (workflow recomandat)
------------------------------------------------------------

Situația 1: adaugi produse noi în site și vrei mapare
1) export din site -> new.xlsx
2) Predict -> rezultat_mapare.xlsx
3) imporți ce are Necesita_verificare = False
4) cele cu True le verifici manual

Situația 2: ai mapat manual / ai listat pe Trendyol produse noi (adevărul)
1) le adaugi în training (labeled_trendyol.xlsx) sau regenerezi exportul din DB
2) rerulezi build_trendyol_taxonomy.py ca să generezi labeled_trendyol_id.xlsx
3) refaci Build (reindex)
4) Predict va fi mai bun

Regula de aur:
- când training-ul se schimbă (mai multe produse mapate corect), refaci Build.

------------------------------------------------------------
10) Troubleshooting (cele mai comune probleme)
------------------------------------------------------------

10.1 Streamlit nu pornește / “streamlit not recognized”
Folosește:
python -m streamlit run app.py

Asigură-te că venv e activ:
.\.venv\Scripts\activate

10.2 Ollama nu răspunde
Deschide:
http://localhost:11434

Sau:
ollama --version

10.3 Eroare “Lipsesc coloane”
Verifică coloanele din Excel:
python -c "import pandas as pd; df=pd.read_excel('labeled_trendyol_id.xlsx'); print(df.columns.tolist())"

Și ajustează text_cols / label_col.

10.4 Build prea lent
- începe cu "Nume,Brand"
- scade workers la 2
- redu descrierea (dacă e foarte lungă)
- testează pe un subset de 300–500 produse ca să validezi flow-ul

10.5 Mapping categorie text -> ID are multe fuzzy_low/none
Înseamnă că textul din export diferă mult față de “category_name” din numele fișierelor.
Soluții:
- normalizezi formatul categoriei din export (separatori, spații)
- verifici manual mapping-ul pentru categoriile problemă (o singură dată)
- crești min_fuzzy_score sau îl scazi, în funcție de cât vrei să fie strict

------------------------------------------------------------
11) Ce urmează (opțional, pasul următor logic)
------------------------------------------------------------

După ce ai CategoryID, următorul nivel este maparea atributelor:
- pentru fiecare CategoryID, deschizi fișierul “<ID> <Category>.xlsx”
- citești lista de caracteristici + valori acceptate
- generezi un fișier de import complet (categorie + atribute populate)

Acest pas se poate automatiza, dar trebuie confirmată structura tabelului din interior.

------------------------------------------------------------
12) Comenzi rapide (rezumat)
------------------------------------------------------------

(în folderul proiectului, venv activ)

1) Activare venv:
.\.venv\Scripts\activate

2) Taxonomie (text -> ID):
python build_trendyol_taxonomy.py --cat_dir "Trendyol Categories" --labeled "labeled_trendyol.xlsx" --category_col "Categorie" --out_catalog "trendyol_categories_catalog.xlsx" --out_mapping "trendyol_text_to_id_mapping.xlsx" --out_labeled_id "labeled_trendyol_id.xlsx" --min_fuzzy_score 90

3) Build (pe CategoryID):
python categorize_engine.py build --labeled labeled_trendyol_id.xlsx --out_dir model_trendyol --text_cols "Nume,Brand,Descriere" --label_col "CategoryID" --workers 4

4) Predict:
python categorize_engine.py predict --input new.xlsx --out_dir model_trendyol --output rezultat_mapare.xlsx --k 15 --min_conf 0.55 --workers 4

5) UI:
python -m streamlit run app.py

------------------------------------------------------------
11) Workflow incremental (nou)
------------------------------------------------------------

Structură store append-only:
- data_store/manifest.json
- data_store/shards/meta_XXXXX.csv + emb_XXXXX.npy
- data_store/hash_index.sqlite (dedup pe content_hash)
- data_store/corrections_gold.csv
- data_store/pseudo_labels.csv
- data_store/review_queue.csv
- data_store/exports/
- data_store/logs/

Paginile UI (multipage Streamlit):
- Dashboard (app.py)
- Ingest
- Mapare
- Review Queue
- Export easySales
- Catalog Manager
- Settings
- Jobs & Logs

Ingest incremental:
- calculează normalize_text + content_hash pe text columns
- dacă hash există în sqlite => duplicat (nu recalculează embedding)
- dacă hash e nou => calculează embedding și creează shard nou
- niciun shard vechi nu este suprascris

Label intern vs export:
- intern: CategoryID
- export easySales: Categoria Text (mapare ID->Text din Catalog Manager)

Gating predict:
- auto-accept dacă trece: AUTO_ACCEPT_CONF + MIN_MARGIN + warm category + validator catalog
- altfel produsul merge în review_queue.csv
- corecțiile umane se salvează în corrections_gold.csv

Retrain batch (fără LLM training):
- python retrain.py --store data_store --mode centroids_only
- python retrain.py --store data_store --mode full
- produce centroids.npz și versiune în data_store/models/model_vXXX.npz

Pornire UI:
- dublu click Start_UI.bat
- sau: python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Task Scheduler (Windows):
- creezi task care rulează periodic:
  .venv\Scripts\python.exe retrain.py --store data_store --mode full
