import type { Turn } from "./api";

export type Example = {
  label: string;
  blurb: string;
  history: Turn[];
  turn: string;
};

export const EXAMPLES: Example[] = [
  {
    label: "Coreference",
    blurb: 'Resolve "he" using prior turns.',
    history: [
      { role: "user", text: "I just finished a book by Isaac Asimov." },
      { role: "bot", text: "He was a prolific science fiction author." },
    ],
    turn: "What is he famous for?",
  },
  {
    label: "Direct lookup",
    blurb: "Specific entity, no history.",
    history: [],
    turn: "Tell me about the Ministry of Magic in Harry Potter.",
  },
  {
    label: "Topic question",
    blurb: "Broad genre overview.",
    history: [],
    turn: "What is the history of science fiction?",
  },
  {
    label: "Theme question",
    blurb: "Concept across multiple articles.",
    history: [],
    turn: "How is time travel depicted in fiction?",
  },
];
