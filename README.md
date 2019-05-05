# ner_brazilian_legal_texts
Named Entity Recognition in Brazilian Legal Texts using Conditional Random Fields and Word Embeddings

# Running

* Download LeNER-Br dataset using the script available in lener directory.

  * ```cd lener && sh download_dataset.sh ```
  
* Run main.py, specifying the NER method and model:

  * ```python main.py --method CRF --model [FirstOrderCRF|HOCRFAD|HOSemiCRFAD]
