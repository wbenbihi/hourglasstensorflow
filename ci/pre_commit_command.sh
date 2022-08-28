#!/bin/bash
pytest tests/
flake8 hourglass_tensorflow --statistics --tee --output-file ./reports/flake8stats.txt
genbadge tests -i reports/junit.xml -o - > reports/tests-badge.svg
genbadge coverage -i reports/coverage.xml -o - > reports/coverage-badge.svg
genbadge flake8 -i reports/flake8stats.txt -o - > reports/flake8-badge.svg
docstr-coverage