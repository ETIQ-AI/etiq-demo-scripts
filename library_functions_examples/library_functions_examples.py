
"""
Example script showing how to use the etiq_copilot library 
- to inspect a python script (user or agent made)
- to capture all the artifacts (no manual instrumentation needed)
- to retrieve: 
    - lineage
    - object/dataframe states
    - model states (if the script includes a model)
    - agent states (if the script includes an agent)

Install the etiq_copilot library; test release:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "etiq-copilot==2.3.0-rc3"
uv pip install --index-strategy unsafe-best-match --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "etiq-copilot==2.3.0-rc3"

Steps shown in this script:
1. Run the etiq copilot scanner on a target Python file in this case `iris_lineage_test.py`.
2. Example of how to generate lineage information from the scan results.
3. Example of how to retrieve the dataframe objects detected by the scanner.
4. Example of a simple verification check to identify which objects are empty.

To try it on your own script point it to the file that initializes the run.
For limitations, see the docs.
"""


from  pathlib import Path

#from etiq_copilot.engine.daemons.utils import working_directory
from etiq_copilot.engine.implementations.scanner.code_scanner import DebuggerCodeScanner
from etiq_copilot.engine.implementations.scanner.scan_results import CodeScannerResult
from verification_functions_codex import get_empty_objects

#Assign your file path 

your_file_path = r"library_functions/iris_lineage_test.py"

def scan_file(
    scan_file_path: Path | str,
) -> CodeScannerResult:
    """Analyze Python code and return scan results.

    This function serves as the main entry point for the code analysis tool.
    """
    scan_file_path = Path(scan_file_path)
    scan_results = CodeScannerResult()
    original_code: str | None = None
    test_scanner = DebuggerCodeScanner()
    try:
        original_code = Path(scan_file_path).read_text(encoding="utf-8")
        #with (
        #    working_directory(scan_file_path.parent),
        #):
        scan_results = test_scanner.scan_code(code_str=original_code)
    except Exception:
        raise
    return scan_results

scan_file = scan_file(scan_file_path=your_file_path)

lineage_graph = scan_file.create_full_lineage_graph()

objects_list = scan_file.list_dataframes()

#print(objects_list)

object_state = scan_file.get_dataframes()

#print(object_state)


#Verification: return empty objects 

empty_objects = get_empty_objects(object_state)

print(empty_objects)

