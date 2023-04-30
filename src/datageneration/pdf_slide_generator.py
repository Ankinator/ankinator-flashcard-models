import jinja2
import os
import pandas as pd
from ankipandas import Collection
import swifter
import cv2
from typing import Union, List
import regex as re
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class PDFSlideGenerator():
    """
    This class converts Anki Cards, which are stored in the local Anki Database into PDF Slides by using different Latex
    Beamer Templates. The Anki cards have to be in the format: [<Front of the card>, <back of the card>].
    Each template has to be in a folder in the specified template folder, and its main file, that is used for generation
    has to be called "main.tex"
    """

    def __init__(self, template_folder="ressources/image-to-text-templates/"):
        self.latex_jinja_env: jinja2.Environment = jinja2.Environment(
            block_start_string='\BLOCK{',
            block_end_string='}',
            variable_start_string='\VAR{',
            variable_end_string='}',
            comment_start_string='\#{',
            comment_end_string='}',
            line_statement_prefix='%%%&',
            line_comment_prefix='%',
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(
                template_folder)
        )
        self.anki_collection: Collection = Collection()
        self.col_media_path: str = os.path.join(self.anki_collection.path.parent.as_posix(), "collection.media")
        self.templates = [f"{template}/main.tex" for template in os.listdir(template_folder)]
        self.template_folder = template_folder

    def _transform_to_landscape(self, image_path: str, soup: BeautifulSoup) -> None:
        image = cv2.imread(image_path)
        if image.shape[0] > image.shape[1]:
            cv2.imwrite(os.path.join(self.col_media_path, soup.find("img").get("src")),
                        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    def _process_images(self, x: List[str]) -> Union[List[str], None]:

        if re.match(r'.*<img src=".*[.](jpg|png)">', x[1]):
            soup = BeautifulSoup(x[1], features="lxml")

            res = []
            for img_tag in soup.find_all("img"):
                full_image_path = os.path.join(self.col_media_path, img_tag.get("src"))
                if os.path.exists(full_image_path) and re.match(r'.*[.](jpg|png)', img_tag.get("src")):
                    self._transform_to_landscape(image_path=full_image_path, soup=soup)

                    res.append(
                        '\includegraphics[width=0.9\\textwidth,height=0.9\\textheight,keepaspectratio]{'
                        + full_image_path + '}')

            if len(res) == 0:
                return None

            return res

        return None

    def _process_answers(self, x: List[str]) -> Union[List[str], None]:
        if re.match(r'.*<br>.*', x[1]):
            inputs = re.split(r'<br>|\n', x[1])
            x = [BeautifulSoup(i, features="lxml").text for i in inputs if
                 not re.match(r'<img src=".*">', i) and BeautifulSoup(i, features="lxml").text != ""]
            if len(x) == 0:
                return None
            return x
        else:
            x = [BeautifulSoup(x[1], features="lxml").text]
            if x[0] != "":
                return x
            else:
                return None

    def generate(self, output_dir: str) -> None:
        df: pd.DataFrame = pd.DataFrame(self.anki_collection.notes.nflds)
        df["answer"] = df.nflds.swifter.apply(lambda x: self._process_answers(x))
        df["images"] = df.nflds.swifter.apply(lambda x: self._process_images(x))

        to_be_dropped: pd.DataFrame = df.loc[(df.answer.isnull()) & (df.images.isnull())]
        logging.info(
            f"Dropping {len(to_be_dropped)} elements,"
            " a list of all dropped elements can be found in the output directory"
        )
        to_be_dropped = pd.concat([to_be_dropped.drop("nflds", axis=1), to_be_dropped.nflds.swifter.apply(pd.Series)])
        to_be_dropped.rename(columns={0: "Question", 1: "Answer"}, inplace=True)
        to_be_dropped.to_csv(os.path.join(output_dir, "dropped.csv"))
        df.drop(to_be_dropped.index, inplace=True)

        for i, template_path in tqdm(enumerate(self.templates)):
            template: jinja2.Template = self.latex_jinja_env.get_template(template_path)
            latex_render: str = template.render(data=df)
            render_path: str = os.path.join(self.template_folder, template_path.replace("main.tex", f"render_{i}.tex"))
            with open(render_path, 'w') as f:
                f.write(latex_render)
            os.system(
                f'cd {render_path.replace(f"render_{i}.tex", "")} && pdflatex -file-line-error'
                + f' -interaction=nonstopmode -synctex=1 -output-format=pdf'
                + f' -output-directory="{os.path.join(os.getcwd(), output_dir)}"'
                + f' render_{i}.tex'
            )


if __name__ == '__main__':
    pdf_generator = PDFSlideGenerator(template_folder='../../ressources/image-to-text-templates/')
    pdf_generator.generate(output_dir='../../out/')
