#!/bin/bash

echo "Move _static"
grep -RiIl '_static' _build | xargs sed -i 's/_static/static/g'
mv _build/html/_static _build/html/static

echo "Move _modules"
grep -RiIl '_modules' _build | xargs sed -i 's/_modules/modules/g'
mv _build/html/_modules _build/html/modules

echo "Move references/_autoapi"
grep -RiIl '_autoapi' _build | xargs sed -i 's/_autoapi/autoapi/g'
mv _build/html/reference/_autoapi _build/html/reference/autoapi

rm -rf _build/html/_sources