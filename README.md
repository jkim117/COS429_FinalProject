# COS429_FinalProject
COS429_Final_Project


1. Git clone this repository: https://github.com/matterport/Mask_RCNN
2. Create new venv: conda create --name cos429final python=3.6
3. Change the requirements.txt file. In particular change the tensorflow line to tensorflow==1.3.0 and the keras line to keras==2.0.8
4. Pip3 install -r requirements.txt
5. Note if you get some error message, try: pip3 --use-feature=2020-resolver install -r requirements.txt 
6. python3 setup.py install
7. Download https://github.com/waleedka/coco and put a copy of this repository within the samples/coco directory
8. Within the PythonAPI folder, run ‘make’
9. Use jupyter-notebook to open up demo.ipynb. Edit the second line of code in the last cell to change the image directory if you want
10. Run the notebook to find masks. r[‘masks’] gives you the masks