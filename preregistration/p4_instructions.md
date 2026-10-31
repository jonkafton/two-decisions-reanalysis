# Labelling instructions

You will read 60 short items. Each has three parts:

- **Evidence (Cue):** a short exchange from early in a long conversation.
- **Question (Trigger):** something the same person said much later in that conversation.
- **Model Prediction:** an AI assistant's reply to that later message. The assistant was meant to remember the earlier exchange.

For each item, decide whether the reply demonstrates awareness of the memory cue in the Evidence, and give it one of two labels. Use only these two definitions, which are the ones the benchmark gives its own judge:

- **correct:** "explicitly acknowledges or adapts to the Memory/Cue (proves recall)"
- **wrong:** "completely ignores the Evidence and gives a generic response."

Every item gets exactly one of the two labels. If neither fits well, choose the one that fits better; there is no third option.

Please work alone, and do not discuss the items with anyone until both of you have finished. Do not search for the items or use an AI tool on them. Do not read the paper these items come from, or its registration, until you have finished. There is no time limit.

Record your labels in the CSV file you were given: one row per item number, with the label written as `correct` or `wrong`.
