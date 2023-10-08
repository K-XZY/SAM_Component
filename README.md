# SAM_Component
A function component utilizing Segment Anything Model with hugging face to segment an image into a list of objects.

## Usage
(For **MacOS** and **Linux** users)

1:
create virtual environment
```
python -m venv .VENVsam
```

activate the virtual environment
```
source .VENVsam/bin/activate
```
To check if this works, do `which pip` and see if it is different from the usual directory.

2:
Install requirements
```
pip install -r requirements.txt
```

3:
Activate Python Interpreter `python`
```
from SAM_function import SAM
```
Select an image from your directory `image_url = 'example.jpg'`

Run the Segmentation Algorithm.
```
Results = SAM(image_url,save_flag = True)
```
When `save_flag` set to `True`, all the segments will be stored in a local folder named `saves`.
