import sys
from collections import Counter
from typing import Counter as CounterT, List

import flutes


class WordCounter(flutes.PoolState):
    def __init__(self):
        self.word_cnt = Counter()

    @flutes.exception_wrapper()
    def count_words(self, sentence: str):
        self.word_cnt.update(word.lower() for word in sentence.split())


def count_words(sentences: List[str]) -> CounterT[str]:
    counter: CounterT[str] = Counter()
    for sent in sentences:
        counter.update(word.lower() for word in sent.split())
    return counter


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [file]")
        sys.exit(1)

    path = sys.argv[1]
    with flutes.work_in_progress("Read file"):
        with open(path) as f:
            sentences = []
            for line in f:
                sentences.append(line)
                if len(sentences) >= 100000:
                    break

    with flutes.work_in_progress("Parallel"):
        with flutes.safe_pool(processes=4, state_class=WordCounter) as pool_stateful:
            for _ in pool_stateful.imap_unordered(WordCounter.count_words, sentences, chunksize=1000):
                pass
            states = pool_stateful.get_states()
        parallel_word_counter: CounterT[str] = Counter()
        for state in states:
            parallel_word_counter.update(state.word_cnt)

    with flutes.work_in_progress("Naive parallel"):
        naive_word_counter: CounterT[str] = Counter()
        data_chunks = flutes.chunk(1000, sentences)
        with flutes.safe_pool(processes=4) as pool:
            for counter in pool.imap_unordered(count_words, data_chunks):
                naive_word_counter.update(counter)

    with flutes.work_in_progress("Sequential"):
        seq_word_counter: CounterT[str] = Counter()
        for sent in sentences:
            seq_word_counter.update(word.lower() for word in sent.split())

    assert seq_word_counter == naive_word_counter == parallel_word_counter


if __name__ == '__main__':
    main()
