import os
import io
import numpy as np
import cv2
import insightface
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
import logging
import hmac, hashlib
from fastapi import Depends, Request


class ImageRequest(BaseModel):
    image_base64: str

class VerifyEmployeeRequest(BaseModel):
    employee_id: str
    image_base64: str

# Add these new request models after your existing ones
class AddEmployeeRequest(BaseModel):
    employee_id: str
    image_base64: str

class DeleteEmployeeRequest(BaseModel):
    employee_id: str

class UpdateEmployeeRequest(BaseModel):
    employee_id: str
    image_base64: str


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
GALLERY_PATH = "Helping Files/Emp_images_nrp.npz"
THRESHOLD = 0.4  # Recognition confidence threshold
USE_GPU = True
# =======================

# Global variables for model (loaded once)
face_app = None
gallery = None
model_loaded = False

# Helper function to save gallery to disk
def save_gallery_to_disk():
    """Save the current gallery state to the .npz file"""
    try:
        logger.info(f"💾 Saving gallery to {GALLERY_PATH}")
        np.savez(GALLERY_PATH, **gallery)
        logger.info(f"✅ Gallery saved successfully with {len(gallery)} employees")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save gallery: {e}")
        return False
    

def load_face_model():
    """Load InsightFace model - CALLED ONLY ONCE"""
    global face_app, model_loaded
    
    if model_loaded:
        logger.info("Model already loaded, skipping...")
        return face_app
    
    logger.info("🔄 Loading face recognition model (one-time setup)...")
    try:
        face_app = insightface.app.FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if USE_GPU else -1
        
        try:
            face_app.prepare(ctx_id=ctx_id)
            logger.info(f"✅ Model loaded successfully on {'GPU' if USE_GPU else 'CPU'}")
        except Exception as e:
            logger.warning(f"⚠️  GPU failed, falling back to CPU: {e}")
            face_app.prepare(ctx_id=-1)
        
        model_loaded = True
        return face_app
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def load_employee_gallery():
    """Load employee embeddings - CALLED ONLY ONCE"""
    global gallery
    
    if gallery is not None:
        logger.info("Gallery already loaded, skipping...")
        return gallery
    
    if not os.path.exists(GALLERY_PATH):
        raise FileNotFoundError(
            f"❌ Gallery file not found: {GALLERY_PATH}\n"
            f"💡 Please create employee embeddings first using generate_embeddings.py"
        )
    
    logger.info(f"🔄 Loading employee gallery from {GALLERY_PATH}...")
    data = np.load(GALLERY_PATH, allow_pickle=True)
    gallery = {k: data[k] for k in data.files}
    logger.info(f"✅ Loaded {len(gallery)} employees from gallery")
    
    return gallery


def verify_employee_with_id(employee_id, embedding, threshold=THRESHOLD):
    """
    Verify a face embedding against a specific employee ID
    Returns: (is_verified: bool, confidence: float, message: str)
    """
    try:
        if gallery is None:
            raise ValueError("Gallery not loaded")

        if employee_id not in gallery:
            return False, 0.0, "Employee ID does not exist"

        stored_embeddings = gallery[employee_id]
        # print(stored_embeddings)

        # Check if embeddings exist
        if stored_embeddings is None or len(stored_embeddings) == 0:
            return False, 0.0, "No stored embeddings for this employee"

        # Compute similarity
        similarities = stored_embeddings @ embedding
        best_score = float(np.max(similarities))
        is_verified = best_score >= threshold

        if not is_verified:
            message = "Face does not match this employee ID"
        else:
            message = "Employee verified successfully"

        return is_verified, best_score, message

    except Exception as e:
        # Catch any unexpected error (empty array, wrong shape, etc.)
        logger.error(f"Verification error for employee {employee_id}: {e}")
        return False, 0.0, f"Verification failed due to error: {str(e)}"


def recognize_face(embedding, threshold=THRESHOLD):
    """
    Recognize a face embedding against the gallery
    
    Args:
        embedding: Face embedding vector
        threshold: Minimum similarity score
        
    Returns:
        tuple: (employee_id, similarity_score) or (None, best_score)
    """
    if gallery is None:
        raise ValueError("Gallery not loaded")
    
    best_score = -1.0
    best_id = None
    
    for employee_id, emb_list in gallery.items():
        # Calculate cosine similarity
        similarities = emb_list @ embedding
        max_sim_idx = np.argmax(similarities)
        max_sim = float(similarities[max_sim_idx])
        
        if max_sim > best_score:
            best_score = max_sim
            best_id = employee_id
    
    return (best_id, best_score) if best_score >= threshold else (None, best_score)

# ==================== HMAC CONFIG ====================

API_KEY = "client_001"
API_SECRET = "sk-proj-faceliveness-87fc416e-d744-44bb-8bb7-f821173e9232"   # 🔐 keep secure
HMAC_TIME_WINDOW = 120  # 2 minutes

# ==================== HMAC AUTH ====================
async def hmac_auth(request: Request) -> None:
    # Get headers
    api_key = request.headers.get("X-API-KEY")
    signature = request.headers.get("X-SIGNATURE")

    # Check all headers present

    if not api_key or not signature:
        logger.warning(
            "Missing auth headers",
            api_key=bool(api_key),
            signature=bool(signature)
        )
        raise HTTPException(status_code=401, detail="Missing authentication headers")

    # Validate API key
    if api_key != API_KEY:
        logger.warning("Invalid API key", provided=api_key)
        raise HTTPException(status_code=401, detail="Invalid API key")

    #Build string to sign (for multipart, don't include body hash)
    # Format: METHOD|PATH
    method = request.method.upper()
    path = request.url.path
    string_to_sign = f"{method}|{path}"

    # Compute expected signature
    expected_signature = hmac.new(
        API_SECRET.encode(),
        string_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()

    # Compare signatures securely
    if not hmac.compare_digest(expected_signature, signature):
        logger.warning(
            "Invalid signature",
            extra={
                "method": method,
                "path": path,
                "expected": expected_signature[:10] + "...",
                "provided": signature[:10] + "..."
            }
        )
        # logger.warning(
        #     "Invalid signature",
        #     method=method,
        #     path=path,
        #     expected=expected_signature[:10] + "...",
        #     provided=signature[:10] + "..."
        #)
        raise HTTPException(status_code=401, detail="Invalid signature")

    #logger.info("HMAC auth successful", api_key=api_key, method=method, path=path)
    logger.info(
    "HMAC auth successful",
    extra={"api_key": api_key, "method": method, "path": path}
)


# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Efficient real-time face recognition system",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model and gallery on server startup - RUNS ONLY ONCE"""
    logger.info("=" * 70)
    logger.info("🚀 Starting Face Recognition API Server")
    logger.info("=" * 70)
    
    try:
        load_face_model()
        load_employee_gallery()
        
        logger.info("=" * 70)
        logger.info("✅ Server ready to accept requests!")
        logger.info(f"📊 Total employees: {len(gallery)}")
        logger.info(f"🎯 Recognition threshold: {THRESHOLD}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error("💡 Please ensure employee_embeddings.npz exists")
        raise


@app.get("/")
async def home():
    """Health check endpoint"""
    return {
        "service": "Face Recognition API",
        "status": "online",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "employees_count": len(gallery) if gallery else 0,
        "threshold": THRESHOLD
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if (model_loaded and gallery is not None) else "unhealthy",
        "model_loaded": model_loaded,
        "gallery_loaded": gallery is not None,
        "employees_count": len(gallery) if gallery else 0
    }


# base64 image endpoint
@app.post("/recognize")
async def recognize_employee(
    payload: ImageRequest,
    _: None = Depends(hmac_auth)
):
    try:
        image_base64 = payload.image_base64.strip()

        # Remove data:image/...;base64,
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        # Fix padding
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += "=" * (4 - missing_padding)

        # Decode
        image = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect faces
        faces = face_app.get(image)
        
        if len(faces) == 0:
            return {
                "employee_id": "unknown",
                "confidence": 0.0,
                "message": "No face detected in image",
                "faces_detected": 0
            }
        
        # Use the first detected face
        face = faces[0]
        embedding = face.normed_embedding
        
        # Recognize
        employee_id, confidence = recognize_face(embedding, THRESHOLD)
        
        if employee_id:
            logger.info(f"✅ Recognized: {employee_id} (confidence: {confidence:.2%})")
            return {
                "employee_id": employee_id,
                "confidence": round(confidence, 4),
                "message": "Employee recognized successfully",
                "faces_detected": len(faces)
            }
        else:
            logger.info(f"❌ Unknown face (best score: {confidence:.2%})")
            return {
                "employee_id": "unknown",
                "confidence": round(confidence, 4),
                "message": f"Face detected but not recognized (below threshold {THRESHOLD})",
                "faces_detected": len(faces)
            }
        
        # return {"status": "image received & decoded successfully"}

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")

    except Exception as e:
        logger.error(f"❌ Recognition error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Recognition failed: {str(e)}"
        )
    


# base64 image endpoint
@app.post("/verify_employee")
async def verify_employee(
    payload: VerifyEmployeeRequest, 
    _: None = Depends(hmac_auth)
):
    employee_id = payload.employee_id
    image_base64 = payload.image_base64

    try:
        if payload is None or not payload.image_base64:
            raise HTTPException(status_code=400, detail="No image provided")

        # ---- Base64 decoding (like recognize endpoint) ----
        image_base64 = payload.image_base64.strip()

        # Remove prefix if exists
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]

        # Fix padding
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += "=" * (4 - missing_padding)

        # Decode to bytes → numpy array → OpenCV image
        image_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # ---- Face detection ----
        faces = face_app.get(img)
        if len(faces) == 0:
            return {
                "employee_id": employee_id,
                "verified": False,
                "confidence": 0.0,
                "message": "No face detected"
            }

        embedding = faces[0].normed_embedding

        # ---- Call verification function ----
        is_verified, confidence, message = verify_employee_with_id(employee_id, embedding)

        return {
            "employee_id": employee_id,
            "verified": is_verified,
            "confidence": round(confidence, 4),
            "message": message
        }

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")

    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")



# 1. ADD EMPLOYEE ENDPOINT
@app.post("/add_employee")
async def add_employee(
    payload: AddEmployeeRequest,
    _: None = Depends(hmac_auth)
):
    """
    Add a new employee with their face embedding to the gallery
    """
    try:
        employee_id = payload.employee_id.strip()
        
        # Check if employee already exists
        if employee_id in gallery:
            raise HTTPException(
                status_code=400, 
                detail=f"Employee {employee_id} already exists. Use /update_employee to modify."
            )
        
        # Decode base64 image
        image_base64 = payload.image_base64.strip()
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += "=" * (4 - missing_padding)
        
        image_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect face and extract embedding
        faces = face_app.get(img)
        
        if len(faces) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No face detected in the provided image"
            )
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
        
        # Get embedding from first face
        embedding = faces[0].normed_embedding
        
        # Store in gallery (as array with single embedding)
        # We wrap it in a list to maintain consistency with the format
        gallery[employee_id] = np.array([embedding])
        
        # Save to disk
        if not save_gallery_to_disk():
            # Rollback if save fails
            del gallery[employee_id]
            raise HTTPException(
                status_code=500, 
                detail="Failed to save employee to disk"
            )
        
        logger.info(f"✅ Added new employee: {employee_id}")
        
        return {
            "status": "success",
            "message": f"Employee {employee_id} added successfully",
            "employee_id": employee_id,
            "embeddings_count": 1,
            "total_employees": len(gallery)
        }
        
    except HTTPException:
        raise
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")
    except Exception as e:
        logger.error(f"❌ Error adding employee: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add employee: {str(e)}"
        )

# 2. DELETE EMPLOYEE ENDPOINT
@app.delete("/delete_employee")
async def delete_employee(
    payload: DeleteEmployeeRequest,
    _: None = Depends(hmac_auth)
):
    """
    Delete an employee and their embeddings from the gallery
    """
    try:
        employee_id = payload.employee_id.strip()
        
        # Check if employee exists
        if employee_id not in gallery:
            raise HTTPException(
                status_code=404,
                detail=f"Employee {employee_id} not found in gallery"
            )
        
        # Store backup in case save fails
        backup_embedding = gallery[employee_id].copy()
        
        # Delete from gallery
        del gallery[employee_id]
        
        # Save to disk
        if not save_gallery_to_disk():
            # Rollback if save fails
            gallery[employee_id] = backup_embedding
            raise HTTPException(
                status_code=500,
                detail="Failed to save changes to disk"
            )
        
        logger.info(f"✅ Deleted employee: {employee_id}")
        
        return {
            "status": "success",
            "message": f"Employee {employee_id} deleted successfully",
            "employee_id": employee_id,
            "total_employees": len(gallery)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting employee: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete employee: {str(e)}"
        )


# 3. UPDATE EMPLOYEE ENDPOINT
@app.put("/update_employee")
async def update_employee(
    payload: UpdateEmployeeRequest,
    _: None = Depends(hmac_auth)
):
    """
    Update an existing employee's face embedding
    """
    try:
        employee_id = payload.employee_id.strip()
        
        # Check if employee exists
        if employee_id not in gallery:
            raise HTTPException(
                status_code=404,
                detail=f"Employee {employee_id} not found. Use /add_employee to create new employee."
            )
        
        # Store backup in case update fails
        backup_embedding = gallery[employee_id].copy()
        
        # Decode base64 image
        image_base64 = payload.image_base64.strip()
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += "=" * (4 - missing_padding)
        
        image_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect face and extract embedding
        faces = face_app.get(img)
        
        if len(faces) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the provided image"
            )
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
        
        # Get new embedding
        new_embedding = faces[0].normed_embedding
        
        # Update gallery with new embedding
        gallery[employee_id] = np.array([new_embedding])
        
        # Save to disk
        if not save_gallery_to_disk():
            # Rollback if save fails
            gallery[employee_id] = backup_embedding
            raise HTTPException(
                status_code=500,
                detail="Failed to save changes to disk"
            )
        
        logger.info(f"✅ Updated employee: {employee_id}")
        
        return {
            "status": "success",
            "message": f"Employee {employee_id} updated successfully",
            "employee_id": employee_id,
            "embeddings_count": 1,
            "total_employees": len(gallery)
        }
        
    except HTTPException:
        raise
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")
    except Exception as e:
        logger.error(f"❌ Error updating employee: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update employee: {str(e)}"
        )


@app.get("/employees")
async def list_employees():
    """Get list of all employees in the system"""
    if gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not loaded")
    
    return {
        "count": len(gallery),
        "employee_ids": sorted(list(gallery.keys()))
    }


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "model_loaded": model_loaded,
        "gallery_loaded": gallery is not None,
        "total_employees": len(gallery) if gallery else 0,
        "threshold": THRESHOLD,
        "using_gpu": USE_GPU
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 FACE RECOGNITION API SERVER")
    print("=" * 70)
    print(f"📁 Gallery file: {GALLERY_PATH}")
    print(f"🎯 Threshold: {THRESHOLD}")
    print(f"💻 GPU: {'Enabled' if USE_GPU else 'Disabled'}")
    print("=" * 70)
    print("\n💡 To use with ngrok:")
    print("   1. Run this server: python api_server.py")
    print("   2. In another terminal: ngrok http 8000")
    print("   3. Use the ngrok URL in your Flutter app")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="info")