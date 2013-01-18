for f in *.py
do
cp __init__.py tmp.py
cat $f >> tmp.py
cp tmp.py $f
rm tmp.py
done