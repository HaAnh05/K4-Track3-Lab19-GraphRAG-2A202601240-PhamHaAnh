import time
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

driver = None

def connect_neo4j():
    """
    Khởi tạo kết nối tới cơ sở dữ liệu Neo4j.
    """
    global driver
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise ValueError("Thiếu cấu hình Neo4j (NEO4J_URI hoặc NEO4J_PASSWORD).")
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    print("✅ Neo4j connected successfully.")
    return driver

def close_neo4j():
    """
    Đóng kết nối driver Neo4j.
    """
    global driver
    if driver is not None:
        try:
            driver.close()
        except Exception:
            pass
        driver = None

def run_cypher(query: str, **params) -> list:
    """
    Thực thi câu lệnh Cypher với retry 3 lần và tự phục hồi kết nối session.
    """
    global driver
    for attempt in range(3):
        try:
            if driver is None:
                connect_neo4j()
            with driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, **params)
                rows = [r.data() for r in result]
                result.consume()
            return rows
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Cypher query failed sau 3 lần thử: {e}\nQuery: {query[:200]}")
            print(f"[Neo4j Retry] Lỗi kết nối lần {attempt+1}: {e}. Đang thử kết nối lại...")
            close_neo4j()
            time.sleep(1.0)

def setup_graph_schema():
    """
    Tạo các constraints và index tối ưu hóa cho Neo4j.
    """
    statements = [
        """
        CREATE CONSTRAINT entity_id IF NOT EXISTS
        FOR (n:Entity) REQUIRE n.id IS UNIQUE
        """,
        """
        CREATE INDEX entity_name_norm IF NOT EXISTS
        FOR (n:Entity) ON (n.name_norm)
        """,
        """
        CREATE INDEX company_name_norm IF NOT EXISTS
        FOR (n:Company) ON (n.name_norm)
        """,
        """
        CREATE INDEX person_name_norm IF NOT EXISTS
        FOR (n:Person) ON (n.name_norm)
        """,
        """
        CREATE INDEX technology_name_norm IF NOT EXISTS
        FOR (n:Technology) ON (n.name_norm)
        """,
    ]
    for stmt in statements:
        run_cypher(stmt)
    print("✅ Neo4j Schema (Constraints & Indexes) ready.")

if __name__ == "__main__":
    connect_neo4j()
    setup_graph_schema()
