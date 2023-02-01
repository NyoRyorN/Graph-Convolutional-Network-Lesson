import glob
import os
from nbconvert import MarkdownExporter

ipynb_path_list = glob.glob("ipynb/*.ipynb")

print(f"{len(ipynb_path_list)}個の.ipynbファイルが見つかりました．")

for ipynb_path in ipynb_path_list:
    filename = os.path.splitext(os.path.basename(ipynb_path))[0]
    print(filename)
    
    markdown_exporter = MarkdownExporter(exclude_output=True)
    markdown, meta = markdown_exporter.from_filename(ipynb_path)

    markdown = markdown.replace("<br>", "")
    
    with open(f"markdown/{filename}.md", "w") as f:
        f.write(markdown)

