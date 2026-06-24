# Embedding Neighbors

A small terminal tool that shows which words GPT-2 considers "closest" to a
word you type. It looks up the word's embedding vector, compares it against
every other word in the vocabulary using cosine similarity, and prints the
nearest neighbors.

Built to understand how token embeddings actually work and what "closeness"
in embedding space really measures.

## What it does

- Loads GPT-2 small (locally, no API)
- Pulls out the word token embedding table (50257 words × 768 numbers each)
- Takes a word, finds its vector, and ranks every other word by cosine
  similarity
- Prints the top neighbors

## Example

```
Enter word: cat

cat
cats
Cat
 cat
 Cat
 cats
```

## The finding

The neighbors aren't what you'd expect. Ask for "cat" and you get *cats*,
*Cat*, and casing/spacing variants. It's not *dog*, *kitten*, or *pet*. Same with
"king": you get *KING*, *King*, and sub-word fragments, not *queen* or *throne*.

That's the real lesson: **GPT-2's raw input embeddings cluster by spelling,
not meaning.** These are the very first lookup, before any attention or
processing happens, so the model has organized them mostly by
words that look alike (not necessarily logically alike).
