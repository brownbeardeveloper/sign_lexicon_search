## Branches

`main` - same as `version_0_1` (stable)

`version_0_1` - version 0.1 branch (faiss & paraphrase-multilingual-MiniLM-L12-v2)

`version_0_2` - version 0.2 branch (faiss & sentence-bert-swedish-cased)

`version_0_3` - version 0.3 branch (ongoing)

## Ranking of BERT Models

1. `version_0_2` - KBLab/sentence-bert-swedish-cased
2. `version_0_1` - SentenceTransformers/paraphrase-multilingual-MiniLM-L12-v2

## Configuration

Check `config.yml` for current configuration and model details.


## To run this vector database

````
mamba --version 
mamba create -f environment.yml
mamba run -n vector_db python main.py
````


## Dataset structure

**Good to know**: To get this project running, you need to download the dataset. Let me know if you need help with that.


Each entry in `dataset/signs.json` represents a sign language word:

```json
{
  "id": "1",
  "word": "taxi",
  "variant_rank": 1,
  "form_description": "D-handen, vänsterriktad och framåtvänd...",
  "main_subject": "fordon",
  "sub_subject": "bilar",
  "english_word": "cab",
  "occurrences_lexicon": 0,
  "media_main_video": "https://example.com/video/1.mp4",
  "media_phrase_videos": [
    {
      "link": "https://example.com/video/2.mp4",
      "example_text": "När jag var liten åkte jag taxi till och från skolan."
    }
  ],
  "scraper_url": "https://example.com/word/1",
  "scraper_fetched_at": "2000-01-01T00:00:00"
},
...
```

| Field                     | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `id`                      | Unique sign ID                                         |
| `word`                    | Swedish word                                           |
| `variant_rank`            | Priority among variants of the same word (1 = highest) |
| `form_description`        | How the sign is performed                              |
| `main_subject`            | Category                                               |
| `sub_subject`             | Subcategory                                            |
| `english_word`            | English translation                                    |
| `media_main_video`        | Video URL of the sign                                  |
| `media_phrase_videos`     | Example sentences with video                           |
