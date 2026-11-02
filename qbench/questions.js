// The questions of qbench: 20 general ones graded by a pattern on the answer, 20 code ones
// graded by running the first fenced block of the answer against tests.
//
// General: { name, q, want } — the answer passes when it matches the regular expression
// `want` (case-insensitive).
// Code: { name, lang, q, tests, ref } — `tests` is appended to the code and run (python3 or
// node); it passes when the script exits 0. `ref` is a correct answer, used by --selftest.
"use strict";

const GENERAL = [
  { name: "capital-france", q: "What is the capital of France?", want: /\bparis\b/ },
  { name: "capital-japan", q: "What is the capital of Japan?", want: /\btokyo\b/ },
  { name: "days-week", q: "How many days are in a week?", want: /\b(7|seven)\b/ },
  { name: "add-12-7", q: "What is 12 plus 7?", want: /\b19\b/ },
  { name: "sky-color", q: "What color is the sky on a clear day?", want: /\bblue\b/ },
  { name: "red-planet", q: "Which planet is known as the Red Planet?", want: /\bmars\b/ },
  { name: "largest-ocean", q: "What is the largest ocean on Earth?", want: /\bpacific\b/ },
  { name: "romeo-juliet", q: "Who wrote Romeo and Juliet?", want: /\bshakespeare\b/ },
  { name: "water-formula", q: "What is the chemical formula of water?", want: /\bh2o\b|h₂o/ },
  { name: "spider-legs", q: "How many legs does a spider have?", want: /\b(8|eight)\b/ },
  { name: "freezing-point", q: "At what temperature in Celsius does water freeze?",
    want: /\b0\b|\bzero\b/ },
  { name: "brazil-language", q: "What language is spoken in Brazil?", want: /\bportuguese\b/ },
  { name: "opposite-hot", q: "What is the opposite of hot?", want: /\bcold\b/ },
  { name: "minutes-hour", q: "How many minutes are in an hour?", want: /\b(60|sixty)\b/ },
  { name: "plants-gas", q: "Which gas do plants take from the air to make their food?",
    want: /carbon dioxide|\bco2\b/ },
  { name: "tallest-mountain", q: "What is the tallest mountain in the world?",
    want: /\beverest\b/ },
  { name: "times-9-3", q: "What is 9 times 3?", want: /\b27\b/ },
  { name: "king-jungle", q: "Which animal is called the king of the jungle?", want: /\blions?\b/ },
  { name: "egypt-continent", q: "On which continent is Egypt?", want: /\bafrica\b/ },
  { name: "plural-mouse", q: "What is the plural of mouse?", want: /\bmice\b/ },
];

const py = (sig, what) => "Write a Python function `" + sig + "` that " + what +
  " Reply with the code in a ```python block.";
const js = (sig, what) => "Write a JavaScript function `" + sig + "` that " + what +
  " Reply with the code in a ```javascript block.";

const CODE = [
  { name: "py-reverse-string", lang: "python", q: py("reverse_string(s)", "returns the string s reversed."),
    tests: 'assert reverse_string("hello") == "olleh"\nassert reverse_string("") == ""\nassert reverse_string("ab c") == "c ba"',
    ref: "def reverse_string(s):\n    return s[::-1]" },
  { name: "py-add", lang: "python", q: py("add(a, b)", "returns the sum of a and b."),
    tests: "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(10, 0) == 10",
    ref: "def add(a, b):\n    return a + b" },
  { name: "py-is-even", lang: "python", q: py("is_even(n)", "returns True when the integer n is even, else False."),
    tests: "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True",
    ref: "def is_even(n):\n    return n % 2 == 0" },
  { name: "py-factorial", lang: "python", q: py("factorial(n)", "returns n! for an integer n >= 0."),
    tests: "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
    ref: "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r" },
  { name: "py-max-of-list", lang: "python", q: py("max_of_list(xs)", "returns the largest number of a non-empty list, without using max()."),
    tests: "assert max_of_list([3, 9, 2]) == 9\nassert max_of_list([-5, -2]) == -2\nassert max_of_list([7]) == 7",
    ref: "def max_of_list(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m" },
  { name: "py-count-vowels", lang: "python", q: py("count_vowels(s)", "returns how many vowels (a, e, i, o, u, upper or lower case) are in s."),
    tests: 'assert count_vowels("hello") == 2\nassert count_vowels("AEIOU xyz") == 5\nassert count_vowels("") == 0',
    ref: 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")' },
  { name: "py-is-palindrome", lang: "python", q: py("is_palindrome(s)", "returns True when s reads the same backwards, else False."),
    tests: 'assert is_palindrome("racecar") is True\nassert is_palindrome("hello") is False\nassert is_palindrome("") is True',
    ref: "def is_palindrome(s):\n    return s == s[::-1]" },
  { name: "py-fizzbuzz", lang: "python", q: py("fizzbuzz(n)", 'returns "Fizz" when n is divisible by 3, "Buzz" when by 5, "FizzBuzz" when by both, else str(n).'),
    tests: 'assert fizzbuzz(3) == "Fizz"\nassert fizzbuzz(10) == "Buzz"\nassert fizzbuzz(15) == "FizzBuzz"\nassert fizzbuzz(7) == "7"',
    ref: 'def fizzbuzz(n):\n    if n % 15 == 0:\n        return "FizzBuzz"\n    if n % 3 == 0:\n        return "Fizz"\n    if n % 5 == 0:\n        return "Buzz"\n    return str(n)' },
  { name: "py-sum-list", lang: "python", q: py("sum_list(xs)", "returns the sum of the numbers in the list xs, without using sum()."),
    tests: "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\nassert sum_list([-4, 4, 5]) == 5",
    ref: "def sum_list(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t" },
  { name: "py-fibonacci", lang: "python", q: py("fibonacci(n)", "returns the n-th Fibonacci number, with fibonacci(0) == 0 and fibonacci(1) == 1."),
    tests: "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
    ref: "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a" },
  { name: "py-c-to-f", lang: "python", q: py("celsius_to_fahrenheit(c)", "converts a temperature from Celsius to Fahrenheit."),
    tests: "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\nassert abs(celsius_to_fahrenheit(-40) + 40) < 1e-9",
    ref: "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32" },
  { name: "py-remove-duplicates", lang: "python", q: py("remove_duplicates(xs)", "returns the list xs without repeated items, keeping the first occurrence of each, in order."),
    tests: "assert remove_duplicates([1, 2, 1, 3, 2]) == [1, 2, 3]\nassert remove_duplicates([]) == []\nassert remove_duplicates(['a', 'a']) == ['a']",
    ref: "def remove_duplicates(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out" },
  { name: "py-capitalize-words", lang: "python", q: py("capitalize_words(s)", "returns s with the first letter of every space-separated word in upper case."),
    tests: 'assert capitalize_words("hello world") == "Hello World"\nassert capitalize_words("a") == "A"\nassert capitalize_words("") == ""',
    ref: 'def capitalize_words(s):\n    return " ".join(w[:1].upper() + w[1:] for w in s.split(" "))' },
  { name: "py-is-prime", lang: "python", q: py("is_prime(n)", "returns True when the integer n is a prime number, else False."),
    tests: "assert is_prime(2) is True\nassert is_prime(17) is True\nassert is_prime(1) is False\nassert is_prime(21) is False",
    ref: "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True" },
  { name: "py-average", lang: "python", q: py("average(xs)", "returns the mean of a non-empty list of numbers."),
    tests: "assert average([1, 2, 3]) == 2\nassert average([5]) == 5\nassert abs(average([1, 2]) - 1.5) < 1e-9",
    ref: "def average(xs):\n    return sum(xs) / len(xs)" },
  { name: "js-reverse-string", lang: "javascript", q: js("reverseString(s)", "returns the string s reversed."),
    tests: 'assert.strictEqual(reverseString("hello"), "olleh");\nassert.strictEqual(reverseString(""), "");',
    ref: 'function reverseString(s) {\n  return s.split("").reverse().join("");\n}' },
  { name: "js-sum-array", lang: "javascript", q: js("sumArray(arr)", "returns the sum of the numbers in the array arr."),
    tests: "assert.strictEqual(sumArray([1, 2, 3]), 6);\nassert.strictEqual(sumArray([]), 0);",
    ref: "function sumArray(arr) {\n  return arr.reduce((a, b) => a + b, 0);\n}" },
  { name: "js-is-even", lang: "javascript", q: js("isEven(n)", "returns true when the integer n is even, else false."),
    tests: "assert.strictEqual(isEven(4), true);\nassert.strictEqual(isEven(7), false);",
    ref: "function isEven(n) {\n  return n % 2 === 0;\n}" },
  { name: "js-max-of", lang: "javascript", q: js("maxOf(arr)", "returns the largest number of a non-empty array."),
    tests: "assert.strictEqual(maxOf([3, 9, 2]), 9);\nassert.strictEqual(maxOf([-5, -2]), -2);",
    ref: "function maxOf(arr) {\n  return Math.max(...arr);\n}" },
  { name: "js-count-words", lang: "javascript", q: js("countWords(s)", "returns the number of words in s, words being separated by spaces."),
    tests: 'assert.strictEqual(countWords("hello big world"), 3);\nassert.strictEqual(countWords("one"), 1);',
    ref: 'function countWords(s) {\n  return s.split(" ").filter((w) => w).length;\n}' },
];

module.exports = { GENERAL, CODE };
