# CBM II - Julia und Felix Abgabe ❤️

## Getting started
```bash
which python
```
If this doesn't say something like 
`/Users/felix-krueckel/default_venv/bin/python
/opt/homebrew/opt/python@3/libexec/bin/python` you should run
`brew install python@3` and ask Gemini how to add it to the top of your path.

```bash
source .venv/bin/activate
```
If this fails run `python -m venv .venv`.
```bash
pip install -r requirements.txt
```

## Best practices
- install packages using `pip install {package}`
- once you installed a package run `pip freeze > requirements.txt`

I'll explain to you how to do this if you are interested, but it's not necessary.