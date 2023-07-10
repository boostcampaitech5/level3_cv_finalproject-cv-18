import uvicorn
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/ocr")
def test(file: bytes = File(...)):
    # ocr결과 리스트
    output = {
        "자동차1": {"coordinate": [(150, 1000), (1500, 2000)], "OCR": "경기5느 2784"},
        "자동차2": {"coordinate": [(300, 450), (1000, 800)], "OCR": "경기51거 2824"},
    }
    return JSONResponse(content=output)


if __name__ == "__main__":
    uvicorn.run(app)
