# In case of old versions pypi will reject upload
# of files with same name
# If this is the case replace dist/* with dist/*-[version_number]*
python3 -m twine upload --repository pypi dist/*