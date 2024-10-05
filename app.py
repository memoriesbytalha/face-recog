from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from chk import process_image
import config  # Import your config file

app = FastAPI()

# Ensure directories exist
os.makedirs(config.input_image_folder, exist_ok=True)
os.makedirs(config.output_image_folder, exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded file
    input_image_path = os.path.join(config.input_image_folder, file.filename)
    with open(input_image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Define the output image path and URL
    output_image_path = os.path.join(config.output_image_folder, file.filename)
    image_url = f"/{config.output_image_folder}/{file.filename}"

    # Process the image
    json_result = process_image(input_image_path, output_image_path, image_url)

    # Return the JSON response
    return JSONResponse(content=json_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
