# Dataset per un RAG didattico Marvel e DC

Questo pacchetto contiene **12 documenti in Markdown**, 6 Marvel e 6 DC, ideati appositamente per esercitarsi con la costruzione di un sistema Retrieval-Augmented Generation.

## Scopo

- Leggere i file e osservare la differenza fra contenuto e metadati.
- Dividere il testo in chunk, sperimentando chunking per paragrafi o per dimensione con overlap.
- Generare embedding e memorizzare chunk, ID, percorso e metadati in un vector database.
- Convertire la domanda in embedding e recuperare i chunk più pertinenti.
- Passare i testi recuperati a un modello linguistico, chiedendo risposte supportate dal contesto e corredate dai percorsi dei documenti.

## Struttura

- `marvel/`: Spider-Man, Iron Man, Captain America, Thor, Black Panther e Avengers.
- `dc/`: Batman, Superman, Wonder Woman, Flash, Green Lantern e Justice League.
- `domande_test.csv`: domande con file attesi per testare il retrieval.

## Convenzioni

Ogni documento contiene un blocco iniziale di **front matter YAML**, racchiuso fra righe `---`, con ID, universo, argomento, fonte e contesto editoriale. Segue il testo con titolo H1 e sezioni H2. Puoi scegliere se includere i metadati nel testo da embeddare oppure salvarli separatamente nel vector DB. Per esempio, `universo` è utile come filtro, `id` e `percorso` sono utili per citare le fonti.

## Continuità e accuratezza

I testi sono schede **originali, sintetiche e orientate all'apprendimento**, non trascrizioni ufficiali né una fonte enciclopedica completa. Privilegiano nozioni introduttive dei fumetti; le continuity e gli adattamenti cinematografici possono differire. Per applicazioni reali, controlla le informazioni contro fonti editoriali affidabili.

## Nota sui diritti

Le schede sono state redatte ex novo a scopo didattico. Nomi e personaggi Marvel e DC appartengono ai rispettivi titolari dei diritti. Questo pacchetto non contiene testi originali dei fumetti né immagini protette.
