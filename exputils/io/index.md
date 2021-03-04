## Input / Output

The input and output code to streamline I/O pipelines or simplify the repetitive loading and saving actions for basic file system use.

### Features

- convenience function that checks if the given filepath exists, and by default, if the filepath exists, appends the datetime to the end of the filepath and creates it, returning the resulting filepath for use in saving the new file.

#### TODO

- Add a convenience read or writer method that prevents the single character error accident of writing over a file to be read.
    Probably just make a read command and use writing as normal with python.
    Then whenever reading occurs, it is simply that command that is called and it has more than a single character difference between writing that will delete the entire file ('r' vs 'w').
