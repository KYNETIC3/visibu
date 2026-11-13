// English for the 250 signs the word model knows (Kaggle asl-signs labels).
//
// The word model recognises single signs. A single word is often not a useful
// message, so common signs also offer short everyday sentences built around
// them. These are fixed, hand-written phrases, offered as suggestions only:
// the signer chooses one (or edits it) before anything is shown.

/** Labels whose plain English differs from the label itself. */
const WORDS: Record<string, string> = {
  TV: 'TV',
  callonphone: 'call on the phone',
  frenchfries: 'french fries',
  glasswindow: 'window',
  haveto: 'have to',
  hesheit: 'he / she / it',
  icecream: 'ice cream',
  minemy: 'my / mine',
  thankyou: 'thank you',
  weus: 'we / us',
  owie: 'it hurts',
  shhh: 'be quiet',
  potty: 'bathroom',
}

/** Everyday sentences for common signs, most likely first. */
const PHRASES: Record<string, string[]> = {
  hello: ['Hello!', 'Hi, how are you doing?'],
  bye: ['Goodbye!', 'See you later.'],
  thankyou: ['Thank you.', 'Thank you very much.'],
  please: ['Please.', 'Could you help me, please?'],
  yes: ['Yes.', 'Yes, that is right.'],
  no: ['No.', 'No, thank you.'],
  fine: ['I am fine.', 'I am fine, thank you.'],
  wait: ['Please wait.', 'Wait a moment, please.'],
  later: ['Later.', 'See you later.'],
  morning: ['Good morning!'],
  night: ['Good night!'],
  hungry: ['I am hungry.', 'Are you hungry?'],
  thirsty: ['I am thirsty.', 'Can I have some water?'],
  water: ['Water, please.', 'Can I have some water?'],
  drink: ['Something to drink, please.', 'Would you like a drink?'],
  food: ['I would like some food.', 'Where can I get food?'],
  sick: ['I feel sick.', 'I am not feeling well.'],
  sleepy: ['I am sleepy.', 'I am tired.'],
  sleep: ['I need to sleep.'],
  nap: ['I need a nap.'],
  happy: ['I am happy.'],
  sad: ['I am sad.'],
  mad: ['I am upset.'],
  hate: ['I do not like it.'],
  like: ['I like it.'],
  where: ['Where?', 'Where is it?'],
  who: ['Who?', 'Who is it?'],
  why: ['Why?'],
  home: ['I want to go home.', 'I am home.'],
  go: ['Let\'s go.', 'I have to go.'],
  stay: ['Please stay.'],
  look: ['Look!', 'Look at this.'],
  listen: ['Listen.'],
  hear: ['I cannot hear you.'],
  finish: ['I am finished.', 'Done.'],
  same: ['Same here.', 'Me too.'],
  now: ['Now.', 'Right now.'],
  tomorrow: ['See you tomorrow.', 'Tomorrow.'],
  yesterday: ['Yesterday.'],
  time: ['What time is it?'],
  potty: ['Where is the bathroom?'],
  owie: ['It hurts.', 'I am hurt.'],
  shhh: ['Please be quiet.'],
  callonphone: ['Please call me.', 'I need to make a phone call.'],
  mom: ['Mom.', 'Where is Mom?'],
  dad: ['Dad.', 'Where is Dad?'],
  quiet: ['Please be quiet.'],
  loud: ['It is too loud.'],
  hot: ['It is hot.'],
  store: ['I am going to the store.'],
  clean: ['It is clean.'],
  dirty: ['It is dirty.'],
  better: ['I feel better.'],
  bad: ['That is bad.'],
  see: ['I see.', 'See you.'],
  think: ['I think so.'],
  talk: ['Can we talk?'],
  read: ['Can you read this?'],
  will: ['I will.'],
  can: ['I can.', 'Can you?'],
  not: ['Not.', 'I do not.'],
  have: ['I have it.'],
  haveto: ['I have to.'],
  minemy: ['That is mine.'],
  yourself: ['Help yourself.'],
}

/** Plain English for a label, e.g. "thankyou" -> "thank you". */
export function signWord(label: string): string {
  return WORDS[label] ?? label
}

const sentenceCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

/** Sentences to suggest for a recognised sign: its everyday phrases, else the word itself. */
export function signPhrases(label: string): string[] {
  return PHRASES[label] ?? [`${sentenceCase(signWord(label))}.`]
}
