def count_word_occurrences():
    word_count = {}
    text = input("Enter the text: ")
    words = text.split()
    for word in words:
        word = word.strip('.,!?-").').lower()
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    sorted_word_count = sorted(word_count.items())
    for word, count in sorted_word_count:
        print(f'{word}: {count}')

count_word_occurrences()
