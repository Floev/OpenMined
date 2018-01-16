#!/bin/sh

UNITY_EXECUTABLE="/Applications/Unity/Unity.app/Contents/MacOS/Unity"
PROJECT_FOLDER="$(pwd)/UnityProject"
RESULTS_FILENAME="results.xml"
RESULTS_FILEPATH="$(pwd)"/$RESULTS_FILENAME
EDITOR_LOG_FILEPATH=" $(echo ~/Library/Logs/Unity/Editor.log)"
GPU_TESTS_FILEPATH=${PROJECT_FOLDER}/Assets/OpenMined.Tests/Editor/*/*TensorGpuTest.cs

echo $PROJECT_FOLDER

## Delete GPU tests file  #FIXME find a way to successfully run GPU tests in Travis CI
rm $GPU_TESTS_FILEPATH

## Run Unity Editor tests
echo "travis_fold:start:editor_tests"
echo "Testing Unity project: $PROJECT_FOLDER"
echo "Running Tests..."

## Run unity tests in background
"$UNITY_EXECUTABLE" \
-projectPath "$PROJECT_FOLDER" \
-runEditorTests \
-editorTestsResultFile "$RESULTS_FILEPATH" \
-batchmode \
-nographics \
&
# Get Unity Process ID
UNITY_PID=$!

echo "Editor log: $EDITOR_LOG_FILEPATH"
# Wait for the tests to begin
sleep 5

# Wait until tests have been executed
while ! [ -f $RESULTS_FILEPATH ]
do
  # Verify if process closed (if it does, it means an unexpected error)
  echo $UNITY_PID
  NUMBER=$(ps aux |grep -i $UNITY_PID|wc -l)
  if [ $NUMBER  -eq 1 ]; then
    echo "travis_fold:end:editor_tests"
    echo "travis_fold:start:editor_log"
    echo "Unity unexpectedly exited, check details below."
    cat $EDITOR_LOG_FILEPATH
    echo "travis_fold:end:editor_log"
    exit 1
  fi
  echo "Haven't finished running tests, trying in 15 seconds"
  sleep 15
done

# Stop Unity process (which is frozen and won't stop by itself)
kill -9 $UNITY_PID
echo "Finished running tests"

echo "Results xml: $RESULTS_FILEPATH"
echo "travis_fold:end:editor_tests"
echo "travis_fold:start:editor_test_results_summary"

# Check if tests failed or were succesfull
if $(grep -q Failed $RESULTS_FILENAME); then
  echo "❌ Editor tests failed."
  echo "Unity exited (see details below)."
else
  echo "✅ Editor tests passed."
  echo "Unity exited with status no errors (see details below)."
fi

echo "travis_fold:end:editor_test_results_summary"
echo "travis_fold:start:editor_test_results_file"

# Print test results as JSON
python \
 scripts/xml-to-json.py \
 -t xml2json \
 "$RESULTS_FILENAME" \
 --pretty \
 --strip_newlines \
 | sed -e $'s/"@/"/g'

echo "travis_fold:end:editor_test_results_file"

# Default exit code
EXIT_CODE=0
# Set exit code 1 if tests failed
if $(grep -q Failed $RESULTS_FILEPATH); then
  EXIT_CODE=1
fi
# Delete xml results file
rm $RESULTS_FILEPATH

exit $EXIT_CODE
