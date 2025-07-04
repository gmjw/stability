# Stability

This package provides the `@stability_test` decorator, which
can be used to easily manage stability tests.


## Background

We often want to wrap our code in what I would call "stability tests".

These tests ensure that nothing has changed about the function(s) being
tested, and often look something like this:

```python
def test_some_func_stability():
    out = some_func()
    expected = pd.DataFrame([[1, 2.2], [3, 4.4]])
    pd.testing.assert_frame_equal(out, expected)
```

But updating tests like this can be time consuming - if
the output of a function has changed, we will have to 
- add a print statement to our test
- copy the new output values
- paste them into our test as the new expected values

This can be a pain, even more so if multiple tests need
updating.


## A better way

The `@stability_test` decorator aims to take all the effort
out of managing stability tests, and also makes their 
implementation quicker and neater. The example test shown
above would simply become:


```python
@stability_test
def test_some_func_stability():
    return some_func()
```

A few notes:
- the test now simply returns the output to be checked
- the `@stability_test` decorator takes care of checking the output vs the expected values 


### How does the library know what the expected values are?

The first time you create a stability test with `@stability_test`
you simply pass in `write=True`.

```python
@stability_test(write=True)
def test_some_func_stability():
    return some_func()
```

When you run the test with `write=True`, the output will be saved to a 
CSV file, in a `resources` sub-folder in the same folder as the test file.

Once this CSV is written, it becomes the expected output of the test,
and whenever the test runs in future the decorator will automatically
compare the test's output (return value) against the contents of the 
CSV 

You can add the CSV file to your repository, and the expected output
of the test is then stored in a clean human-readable format. Any future
changes in the output of the function will be clearly visible in the 
diff on the CSV file. And it only takes adding `write=True` and 
re-running the test to upate the values in the CSV - no adding
print statements, no copying or pasting.

NOTE: Remember to _remove_ `write=True` once the CSV file has been
written or updated - this is enforced by the test raising an error
a `RememberError`) once it has written new values to disk.


### Checking multiple outputs in the same test

The decorator can also be used to check values which are 
yielded from a test set up as a generator function. For example:

```python
@stability_test
def test_some_func_stability_two_cases():
    yield some_func(1, 'a')
    yield some_func(2, 'b')
```

This test will compare the outputs of each function call to 
the contents of an associated CSV file. Both files can be updated at 
once by running the test with `write=True`, as discussed above.


### Updating outputs for multiple tests at once

The decorator is most useful when updating multiple expected values
(across multiple tests). Rather than repeating the work of adding 
`write=True` to each test individually, you can set the environment
variable

```commandline
STABILITY_WRITE_DATA=1  # any True-like string is fine
```

Then run any tests for which you want to update expected values. 
The decorator will take care of the rest - all expected output CSVs
will be updated. You just need to add/commit the changed CSV files 
to git or whichever VCS you are using (and remember to un-set the
environment variable when you're done).
