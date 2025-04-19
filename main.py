from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from database import get_connection
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

embeddings = np.load("embeddings.npy")
df = pd.read_pickle("medicine_data2.pkl")

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

class MedicineQuery(BaseModel):
    query: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace '*' with your frontend IP/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {"message" : " hello from the fastapi"}

@app.post("/search_medicine")
async def search_medicine(query: MedicineQuery):
    try:
        query_embedding = model.encode(query.query)
        
        similarities = util.cos_sim(query_embedding, embeddings)[0]
        
        best_match_idx = similarities.argmax().item()
        
        medicine_details = df.iloc[best_match_idx].to_dict()

        return {
            "status": "success",
            "data": medicine_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

class InventoryItem(BaseModel):
    id: int
    medicine_name: str
    details: str
    stock: int
    price: float
    expiry_date: date

class SalesItem(BaseModel):
    sale_id: int
    medicine_id: int
    quantity: int
    sale_date: datetime
    total_price: float

@app.get('/inventory')
def read_inventory():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM inventory")
    result = cursor.fetchall()

    cursor.close()
    conn.close()
    return result

@app.get('/sales')
def read_sales():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("select * from sales")
    result = cursor.fetchall()

    cursor.close()
    conn.close()
    return result

@app.get('/inventory/low-stock')
def get_low_stock_items(limit: int = 5):
    """Get medicines with lowest stock (top 5 by default)"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, medicine_name, stock, price, expiry_date, details FROM inventory ORDER BY stock ASC LIMIT %s",
            (limit,)
        )
        low_stock_items = cursor.fetchall()
        
        return {
            "low_stock_items": low_stock_items,
            "count": len(low_stock_items)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching low stock items: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.get('/inventory/expiring-soon')
def get_expiring_soon(limit: int = 5, days: int = 30):
    """Get medicines expiring within the next X days (default: 30 days)"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT id, medicine_name, stock, price, expiry_date, details 
            FROM inventory 
            WHERE expiry_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL %s DAY)
            ORDER BY expiry_date ASC
            LIMIT %s
            """,
            (days, limit)
        )
        expiring_items = cursor.fetchall()
        
        return {
            "expiring_soon": expiring_items,
            "count": len(expiring_items),
            "days_threshold": days
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching expiring items: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.post('/sales/create')
def create_sale(sale_data: dict):
    """Create a new sale and update inventory"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Start transaction
        conn.start_transaction()
        
        # 1. Insert sale record
        cursor.execute(
            """
            INSERT INTO sales (medicine_id, quantity, sale_date, total_price)
            VALUES (%s, %s, %s, %s)
            """,
            (
                sale_data['medicine_id'],
                sale_data['quantity'],
                datetime.now(),
                sale_data['total_price']
            )
        )
        
        # 2. Update inventory stock
        cursor.execute(
            """
            UPDATE inventory 
            SET stock = stock - %s 
            WHERE id = %s
            """,
            (sale_data['quantity'], sale_data['medicine_id'])
        )
        
        # Commit transaction
        conn.commit()
        
        return {"success": True, "message": "Sale recorded successfully"}
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing sale: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.get('/inventory/search')
def search_inventory(query: str):
    """Search medicines by name"""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT id, medicine_name, stock, price 
            FROM inventory 
            WHERE medicine_name LIKE %s
            LIMIT 10
            """,
            (f"%{query}%",)
        )
        results = cursor.fetchall()
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error searching inventory: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()