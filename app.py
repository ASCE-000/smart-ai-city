
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import deque, defaultdict, Counter
import heapq, random, math, sqlite3, time
from pathlib import Path

st.set_page_config(page_title="Smart City AI | Intelligence Hub", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme / UI ----------
st.markdown("""
<style>
:root { --navy:#101828; --muted:#667085; --line:#e4e7ec; --blue:#356df3; }
.block-container {padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#101828 0%,#172b4d 100%);}
[data-testid="stSidebar"] * {color: #f2f4f7 !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding: .35rem .55rem; border-radius: 8px;}
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#111c35,#234ca0); color:white; margin-bottom:1.2rem;}
.hero h1 {font-size:2.25rem; margin:0 0 .4rem 0; color:white;}
.hero p {color:#dbe6ff; font-size:1.02rem; margin:0;}
.kpi {border:1px solid var(--line); padding:1rem 1.1rem; border-radius:14px; background:white;}
.kpi .label {color:#667085; font-size:.86rem;} .kpi .value {font-size:1.55rem; font-weight:700; color:#101828;}
.section-note {color:#667085; font-size:.92rem;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #eaecf0; padding:12px 16px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

PAGES = [
    "Overview", "AI Problem Formulation", "Emergency Route Planner", "Warehouse Robot",
    "Hotel Knowledge Base", "Customer Support LLM", "Traffic Controller", "Pricing Game",
    "Smart Irrigation", "Credit Risk Fuzzy Systems", "Customer Churn", "Customer Segmentation",
    "Algorithm Lab", "Documentation & Viva"
]
page = st.sidebar.radio("NAVIGATION", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("B.Tech AI & Data Science • Local demo")
st.sidebar.caption("Synthetic data is labelled throughout.")

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def fuzzy_tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x-a)/(b-a) if b != a else 1.0
    return (c-x)/(c-b) if c != b else 1.0

def route_graph():
    return {
        "Hospital": {"Central": 3, "Market": 5},
        "Central": {"Hospital": 3, "Station": 4, "College": 2},
        "Market": {"Hospital": 5, "Station": 3, "Airport": 8},
        "Station": {"Central": 4, "Market": 3, "Airport": 6, "College": 3},
        "College": {"Central": 2, "Station": 3, "Airport": 7},
        "Airport": {"Market": 8, "Station": 6, "College": 7},
    }

def bfs(graph, start, goal):
    q=deque([(start,[start])]); seen={start}; expanded=[]
    while q:
        node,path=q.popleft(); expanded.append(node)
        if node==goal: return path, expanded
        for nxt in graph.get(node,{}):
            if nxt not in seen: seen.add(nxt); q.append((nxt,path+[nxt]))
    return [], expanded

def dfs(graph, start, goal):
    stack=[(start,[start])]; seen=set(); expanded=[]
    while stack:
        node,path=stack.pop()
        if node in seen: continue
        seen.add(node); expanded.append(node)
        if node==goal: return path, expanded
        for nxt in reversed(list(graph.get(node,{}))):
            if nxt not in seen: stack.append((nxt,path+[nxt]))
    return [], expanded

def ucs(graph, start, goal):
    pq=[(0,start,[start])]; best={start:0}; expanded=[]
    while pq:
        cost,node,path=heapq.heappop(pq)
        if cost != best.get(node): continue
        expanded.append(node)
        if node==goal: return path,cost,expanded
        for nxt,w in graph.get(node,{}).items():
            nc=cost+w
            if nc<best.get(nxt,float("inf")):
                best[nxt]=nc; heapq.heappush(pq,(nc,nxt,path+[nxt]))
    return [],float("inf"),expanded

def grid_astar(grid, start, goal):
    rows,cols=grid.shape
    h=lambda p: abs(p[0]-goal[0])+abs(p[1]-goal[1])
    pq=[(h(start),0,start,[start])]; best={start:0}; expanded=[]
    while pq:
        _,cost,p,path=heapq.heappop(pq)
        if cost!=best.get(p): continue
        expanded.append(p)
        if p==goal:return path,cost,expanded
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            q=(p[0]+dr,p[1]+dc)
            if 0<=q[0]<rows and 0<=q[1]<cols and grid[q]==0:
                nc=cost+1
                if nc<best.get(q,float("inf")):
                    best[q]=nc;heapq.heappush(pq,(nc+h(q),nc,q,path+[q]))
    return [],float("inf"),expanded

def draw_grid(grid, path, expanded, start, goal):
    z=np.zeros((*grid.shape,3),dtype=int)+245
    z[grid==1]=[45,55,72]
    for r,c in expanded:
        if grid[r,c]==0:z[r,c]=[194,218,255]
    for r,c in path:z[r,c]=[53,109,243]
    z[start]=[16,185,129];z[goal]=[245,158,11]
    fig=go.Figure(go.Heatmap(z=np.arange(grid.size).reshape(grid.shape),showscale=False,hoverinfo="skip",colorscale=[[0,"#f5f7fb"],[0.25,"#c2daff"],[0.5,"#356df3"],[0.75,"#2d3748"],[1,"#10b981"]]))
    # draw cell colors as shapes for predictable palette
    fig=go.Figure()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color="#f5f7fb"
            if grid[r,c]==1: color="#344054"
            if (r,c) in expanded: color="#c2daff"
            if (r,c) in path: color="#356df3"
            if (r,c)==start: color="#10b981"
            if (r,c)==goal: color="#f59e0b"
            fig.add_shape(type="rect",x0=c,y0=r,x1=c+1,y1=r+1,fillcolor=color,line=dict(color="#d0d5dd",width=1))
    fig.update_yaxes(autorange="reversed",range=[grid.shape[0],0],showticklabels=False)
    fig.update_xaxes(range=[0,grid.shape[1]],showticklabels=False)
    fig.update_layout(height=430,margin=dict(l=5,r=5,t=5,b=5),plot_bgcolor="white")
    return fig

def init_db():
    conn=sqlite3.connect("smart_city.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS rooms(room TEXT PRIMARY KEY, type TEXT, price INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS bookings(id INTEGER PRIMARY KEY AUTOINCREMENT, guest TEXT, room TEXT, nights INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    if conn.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]==0:
        conn.executemany("INSERT INTO rooms VALUES(?,?,?)",[("101","Standard",1800),("102","Standard",1800),("201","Deluxe",2800),("202","Deluxe",2800),("301","Suite",4500)])
    conn.commit()
    return conn

if page=="Overview":
    hero("AI-Powered Smart City Intelligence System","One academic dashboard. Twelve practical concepts. Real algorithms, transparent demo data.")
    st.markdown("### City intelligence at a glance")
    cols=st.columns(4)
    for col, val, label in zip(cols,["12","11","Local","SQLite"],["Interactive modules","Core practical areas","No paid API keys","Persistent bookings"]):
        with col: st.metric(label,val)
    st.markdown("")
    c1,c2=st.columns([1.25,1])
    with c1:
        st.markdown("#### Explore the platform")
        st.write("A unified lab for search, reasoning, language modelling, fuzzy systems, adversarial search and learning algorithms.")
        st.info("Use the sidebar to open a module. Most modules include editable inputs and live outputs.")
        st.markdown("- **Search:** BFS, DFS, UCS and A*")
        st.markdown("- **Reasoning:** PEAS formulation, rule-based booking")
        st.markdown("- **Intelligent control:** neuro-fuzzy traffic and irrigation")
        st.markdown("- **Learning:** Perceptron churn model and competitive customer segmentation")
    with c2:
        st.markdown("#### Practical coverage")
        df=pd.DataFrame({"Area":["Search","Fuzzy systems","Learning","Reasoning","Language models","Game search"],"Modules":[3,3,2,2,1,1]})
        st.plotly_chart(px.bar(df,x="Modules",y="Area",orientation="h",color="Modules",title="Module coverage"),use_container_width=True)
    st.markdown("#### Important project note")
    st.warning("This is an educational prototype. Traffic, credit-risk, customer, and route examples use synthetic or illustrative data. It does not connect to emergency services, financial institutions, or physical city infrastructure.")

elif page=="AI Problem Formulation":
    hero("AI Problem Formulation","PEAS and state-space design for an emergency transportation agent")
    a,b=st.columns(2)
    with a:
        st.markdown("#### PEAS framework")
        st.table(pd.DataFrame({"Component":["Performance measure","Environment","Actuators","Sensors"],"Definition":["Minimise response time and route cost","Road network, traffic, hospitals","Select next road segment / route","GPS, road status, estimated travel cost"]}))
    with b:
        st.markdown("#### Problem definition")
        st.write("**Initial state:** ambulance at Central.  **Goal state:** Hospital.")
        st.write("**Actions:** travel along a connected road.")
        st.write("**Transition model:** moving across an edge reaches the adjacent node.")
        st.write("**Goal test:** current node equals Hospital.")
        st.write("**Path cost:** sum of road-cost weights.")
    g=route_graph()
    edges=[(u,v,w) for u,adj in g.items() for v,w in adj.items() if u<v]
    coords={"Hospital":(0,1),"Central":(1,2),"Market":(2,0),"Station":(2,2),"College":(3,3),"Airport":(4,1)}
    fig=go.Figure()
    for u,v,w in edges:
        x1,y1=coords[u];x2,y2=coords[v]
        fig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode="lines+text",text=["",str(w)],textposition="middle center",line=dict(color="#98a2b3",width=2),showlegend=False,hoverinfo="skip"))
    for n,(x,y) in coords.items():
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[n],textposition="top center",marker=dict(size=22,color="#356df3"),showlegend=False))
    fig.update_layout(title="Illustrative city road graph (edge labels = cost)",height=390,showlegend=False,margin=dict(l=5,r=5,t=35,b=5),xaxis=dict(visible=False),yaxis=dict(visible=False))
    st.plotly_chart(fig,use_container_width=True)

elif page=="Emergency Route Planner":
    hero("Emergency Route Planner","Compare uninformed search and cost-aware route planning")
    g=route_graph(); nodes=list(g)
    c1,c2,c3=st.columns(3)
    start=c1.selectbox("Start location",nodes,index=0)
    goal=c2.selectbox("Destination",nodes,index=5)
    method=c3.selectbox("Highlight route",["Uniform Cost Search","BFS","DFS"])
    results=[]
    for name,fn in [("BFS",bfs),("DFS",dfs)]:
        p,e=fn(g,start,goal); cost=sum(g[u][v] for u,v in zip(p,p[1:])) if p else 0
        results.append({"Algorithm":name,"Route":" → ".join(p) if p else "No route","Edges":max(0,len(p)-1),"Cost":cost,"Nodes explored":len(e)})
    p,cost,e=ucs(g,start,goal)
    results.append({"Algorithm":"UCS","Route":" → ".join(p) if p else "No route","Edges":max(0,len(p)-1),"Cost":cost if p else "—","Nodes explored":len(e)})
    st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True)
    if method=="BFS": path,explored=bfs(g,start,goal); cost=sum(g[u][v] for u,v in zip(path,path[1:])) if path else 0
    elif method=="DFS": path,explored=dfs(g,start,goal); cost=sum(g[u][v] for u,v in zip(path,path[1:])) if path else 0
    else: path,cost,explored=ucs(g,start,goal)
    c1,c2,c3=st.columns(3)
    c1.metric("Path cost",cost if path else "Unreachable")
    c2.metric("Road segments",max(0,len(path)-1))
    c3.metric("Nodes explored",len(explored))
    st.success(" → ".join(path) if path else "No path exists.")
    st.caption("BFS minimises edge count on an unweighted graph; UCS minimises total non-negative edge cost. These may produce different routes.")

elif page=="Warehouse Robot":
    hero("Warehouse Robot","A* pathfinding with editable obstacles")
    st.write("Green = start • Orange = goal • Blue = route • Light blue = explored • Dark = obstacle")
    default=np.zeros((10,12),dtype=int)
    for r,c in [(1,3),(2,3),(3,3),(4,3),(5,3),(6,6),(6,7),(6,8),(2,8),(3,8),(4,8)]: default[r,c]=1
    if "grid" not in st.session_state: st.session_state.grid=default.copy()
    grid=st.session_state.grid
    left,right=st.columns([1,1.35])
    with left:
        st.markdown("#### Edit map")
        rr=st.number_input("Obstacle row (0–9)",0,9,1)
        cc=st.number_input("Obstacle column (0–11)",0,11,1)
        x,y=st.columns(2)
        if x.button("Add obstacle",use_container_width=True): grid[int(rr),int(cc)]=1; st.session_state.grid=grid
        if y.button("Remove obstacle",use_container_width=True): grid[int(rr),int(cc)]=0; st.session_state.grid=grid
        if st.button("Reset warehouse",use_container_width=True): st.session_state.grid=default.copy(); grid=st.session_state.grid
        st.caption("Start: (0, 0) • Goal: (9, 11)")
    start=(0,0); goal=(9,11); grid[start]=0; grid[goal]=0
    path,cost,expanded=grid_astar(grid,start,goal)
    with right:
        st.plotly_chart(draw_grid(grid,path,expanded,start,goal),use_container_width=True)
    x,y,z=st.columns(3)
    x.metric("Path cost",cost if path else "Unreachable")
    y.metric("Expanded nodes",len(expanded))
    z.metric("Path cells",len(path))
    if not path: st.error("Destination is unreachable with the current obstacles.")
    else: st.success("A* found a route using Manhattan distance.")

elif page=="Hotel Knowledge Base":
    hero("Hotel Knowledge Base","Rule-based room availability and persistent booking records")
    conn=init_db()
    rooms=pd.read_sql_query("SELECT * FROM rooms",conn)
    booked=pd.read_sql_query("SELECT room FROM bookings",conn)["room"].tolist()
    rooms["Status"]=rooms["room"].apply(lambda r:"Booked" if r in booked else "Available")
    st.dataframe(rooms,use_container_width=True,hide_index=True)
    left,right=st.columns(2)
    with left:
        st.markdown("#### Make a booking")
        guest=st.text_input("Guest name")
        room=st.selectbox("Room",rooms["room"].tolist())
        nights=st.number_input("Number of nights",1,30,1)
        if st.button("Book room",type="primary"):
            is_booked=conn.execute("SELECT COUNT(*) FROM bookings WHERE room=?",(room,)).fetchone()[0]>0
            if not guest.strip(): st.error("Enter a guest name.")
            elif is_booked: st.error("That room is already booked.")
            else:
                conn.execute("INSERT INTO bookings(guest,room,nights) VALUES(?,?,?)",(guest.strip(),room,int(nights)));conn.commit()
                st.success(f"Room {room} booked for {guest}.");st.rerun()
    with right:
        st.markdown("#### Current bookings")
        bookings=pd.read_sql_query("SELECT id,guest,room,nights,created_at FROM bookings ORDER BY id DESC",conn)
        if bookings.empty: st.info("No bookings yet.")
        else:
            st.dataframe(bookings,use_container_width=True,hide_index=True)
            booking_id=st.selectbox("Booking ID to cancel",bookings["id"].tolist())
            if st.button("Cancel booking"):
                conn.execute("DELETE FROM bookings WHERE id=?",(int(booking_id),));conn.commit();st.success("Booking cancelled.");st.rerun()
    st.caption("This module uses explicit Python rules and SQLite. It does not execute PROLOG; SWI-Prolog integration is a documented extension.")

elif page=="Customer Support LLM":
    hero("Customer Support Text Assistant","Tiny statistical language models: Bigram µLM and Trigram SLM")
    corpus=[
      "your order is being processed","your order is out for delivery","your delivery is delayed",
      "refund requests are reviewed within five days","payment failed please try another payment method",
      "update your account email in account settings","track your order using the tracking number",
      "your refund has been approved","delivery status is updated every hour","contact support for account problems",
      "please verify your payment details","your order has been delivered"
    ]
    tokens=[["<s>"]+s.lower().split()+["</s>"] for s in corpus]
    bi=defaultdict(Counter); tri=defaultdict(Counter)
    for t in tokens:
        for i in range(len(t)-1): bi[t[i]][t[i+1]]+=1
        for i in range(len(t)-2): tri[(t[i],t[i+1])][t[i+2]]+=1
    context=st.text_input("Enter a word or two-word context","your order")
    words=context.lower().split()
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Bigram µLM")
        opts=bi[words[-1]] if words else Counter()
        if opts:
            nextw,n=opts.most_common(1)[0]; st.success(f"Next word prediction: **{nextw}**")
            st.caption(f"Observed transition count: {n}")
        else: st.info("Unknown context. Try: order, delivery, refund, payment.")
    with c2:
        st.markdown("#### Trigram SLM")
        key=tuple(words[-2:]) if len(words)>=2 else ()
        opts=tri[key] if key else Counter()
        if opts:
            nextw,n=opts.most_common(1)[0]; st.success(f"Next word prediction: **{nextw}**")
            st.caption(f"Observed sequence count: {n}")
        else: st.info("No trigram match for this context.")
    if st.button("Generate short continuation"):
        current=words[:]
        for _ in range(8):
            nxts=tri[tuple(current[-2:])] if len(current)>=2 else bi[current[-1]] if current else Counter()
            if not nxts: break
            current.append(nxts.most_common(1)[0][0])
            if current[-1]=="</s>": break
        st.write(" ".join(w for w in current if w not in ("<s>","</s>")))

elif page=="Traffic Controller":
    hero("Neuro-Fuzzy Traffic Controller","Synthetic traffic dataset + interpretable fuzzy recommendation")
    rng=np.random.default_rng(42)
    df=pd.DataFrame({"Vehicle count":rng.integers(5,100,240),"Average speed":rng.uniform(5,65,240),"Road occupancy":rng.uniform(5,100,240)})
    df["Synthetic green time"]=np.clip(15+0.55*df["Vehicle count"]-0.28*df["Average speed"]+0.18*df["Road occupancy"],10,120)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    X=df[["Vehicle count","Average speed","Road occupancy"]]; y=df["Synthetic green time"]
    model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,8),max_iter=1200,random_state=42))
    model.fit(X,y)
    v=st.slider("Vehicle count",5,100,55); speed=st.slider("Average speed (km/h)",5,65,30); occ=st.slider("Road occupancy (%)",5,100,60)
    ann=float(model.predict(pd.DataFrame([[v,speed,occ]],columns=X.columns))[0])
    congestion=np.clip(0.45*v/100+0.25*(65-speed)/60+0.30*occ/100,0,1)
    fuzzy=25+85*congestion
    final=0.65*ann+0.35*fuzzy
    c1,c2,c3=st.columns(3);c1.metric("ANN estimate",f"{ann:.1f} s");c2.metric("Fuzzy output",f"{fuzzy:.1f} s");c3.metric("Combined recommendation",f"{final:.1f} s")
    st.caption("Synthetic training target; weighted blend = 65% ANN + 35% fuzzy congestion score. Not connected to real traffic lights.")
    st.plotly_chart(px.scatter(df,x="Vehicle count",y="Synthetic green time",color="Road occupancy",title="Synthetic training data"),use_container_width=True)

elif page=="Pricing Game":
    hero("E-commerce Pricing Game","Two-player adversarial search using Minimax and alpha-beta pruning")
    prices=["Low price","Medium price","High price"]
    utility=np.array([[2,5,7],[1,4,6],[-2,2,5]])
    st.write("Illustrative payoff matrix for Seller A. Seller A maximises utility; Seller B selects a response that minimises it.")
    st.dataframe(pd.DataFrame(utility,index=[f"A: {p}" for p in prices],columns=[f"B: {p}" for p in prices]),use_container_width=True)
    def minimax(depth, maximizing, alpha, beta, row=0, nodes=None):
        if nodes is None:nodes={"minimax":0,"ab":0}
        if depth==2:return (int(utility[row,0]),None,nodes)
        if maximizing:
            best=-math.inf; action=None
            for i in range(3):
                nodes["minimax"]+=1
                val,_,nodes=minimax(depth+1,False,alpha,beta,i,nodes)
                if val>best:best=val;action=i
            return best,action,nodes
        best=math.inf
        for j in range(3):
            nodes["minimax"]+=1
            val=int(utility[row,j])
            best=min(best,val)
        return best,row,nodes
    # Correct 2-ply minimax over payoff matrix, and alpha-beta node accounting.
    mm_values=[min(utility[i,:]) for i in range(3)]
    mm_action=int(np.argmax(mm_values)); mm_value=int(mm_values[mm_action])
    def ab_search():
        alpha=-math.inf; beta=math.inf; nodes=0; best=-math.inf; best_i=0
        for i in range(3):
            worst=math.inf
            for j in range(3):
                nodes+=1; val=int(utility[i,j]); worst=min(worst,val); beta=min(beta,worst)
                if beta<=alpha: break
            if worst>best:best=worst;best_i=i
            alpha=max(alpha,best)
        return best_i,best,nodes
    ab_i,ab_v,ab_nodes=ab_search()
    c1,c2,c3=st.columns(3);c1.metric("Minimax choice",prices[mm_action]);c2.metric("Utility",mm_value);c3.metric("Alpha-beta leaves checked",ab_nodes)
    st.caption("The payoff matrix is illustrative, not market data. Alpha-beta pruning can skip branches that cannot change the final choice.")

elif page=="Smart Irrigation":
    hero("Smart Irrigation","Mamdani-style fuzzy rules for a water-pump recommendation")
    moisture=st.slider("Soil moisture (%)",0,100,35); temp=st.slider("Air temperature (°C)",0,50,32)
    dry=np.clip((60-moisture)/60,0,1); hot=np.clip(temp/45,0,1)
    duration=float(np.clip(2+12*dry+6*hot,0,25))
    baseline=float(20 if moisture<25 else 12 if moisture<40 else 5 if moisture<60 else 0)
    c1,c2=st.columns(2);c1.metric("Fuzzy recommendation",f"{duration:.1f} minutes");c2.metric("Threshold baseline",f"{baseline:.1f} minutes")
    st.write("**Rule examples**")
    st.markdown("- IF soil is dry AND temperature is high THEN pump duration is long.")
    st.markdown("- IF soil is medium AND temperature is medium THEN pump duration is moderate.")
    st.markdown("- IF soil is wet THEN pump duration is short or zero.")
    xs=np.linspace(0,100,101)
    mf=pd.DataFrame({"Soil moisture":xs,"Low (dry)":np.maximum(0,(60-xs)/60),"Medium":np.maximum(0,1-np.abs(xs-50)/30),"High (wet)":np.maximum(0,(xs-40)/60)})
    st.plotly_chart(px.line(mf,x="Soil moisture",y=["Low (dry)","Medium","High (wet)"],title="Illustrative soil-moisture membership functions"),use_container_width=True)
    st.warning("Recommendation only; no physical pump or irrigation hardware is connected.")

elif page=="Credit Risk Fuzzy Systems":
    hero("Credit Risk Fuzzy Analysis","Compare educational Mamdani and Sugeno outputs")
    score=st.slider("Credit score",300,850,680); dti=st.slider("Debt-to-income ratio (%)",0,100,35)
    risk=np.clip((700-score)/400*0.65+dti/100*0.8,0,1)
    mamdani=100*risk
    # Sugeno singleton consequents, weighted by illustrative rules
    low=max(0,min(1,(score-550)/150)); high=max(0,min(1,(700-score)/150))
    dti_high=max(0,min(1,(dti-30)/40)); dti_low=1-dti_high
    weights=np.array([low*dti_low, low*dti_high, high*dti_low, high*dti_high])
    consequents=np.array([15,40,55,90],dtype=float)
    sugeno=float(np.dot(weights,consequents)/weights.sum()) if weights.sum() else 50.0
    c1,c2=st.columns(2);c1.metric("Mamdani-style risk score",f"{mamdani:.1f}/100");c2.metric("Sugeno weighted output",f"{sugeno:.1f}/100")
    st.caption("These are simplified educational fuzzy-style calculations, not a validated lending model or a real credit decision.")
    st.markdown("**Interpretation:** higher illustrative risk scores indicate a higher risk signal within this demonstration only.")

elif page=="Customer Churn":
    hero("Customer Churn","Perceptron classifier trained on reproducible synthetic customers")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    rng=np.random.default_rng(23); n=600
    df=pd.DataFrame({"Engagement":rng.uniform(0,100,n),"Monthly usage":rng.uniform(0,80,n),"Support tickets":rng.integers(0,8,n),"Satisfaction":rng.uniform(1,5,n)})
    prob=1/(1+np.exp(-(-1.0-0.035*df["Engagement"]-0.025*df["Monthly usage"]+0.32*df["Support tickets"]-0.45*df["Satisfaction"]+2.4)))
    df["Churn"]=(rng.random(n)<prob).astype(int)
    from sklearn.linear_model import Perceptron
    X=df.drop(columns="Churn"); y=df["Churn"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    scaler=StandardScaler(); Xtr_s=scaler.fit_transform(Xtr);Xte_s=scaler.transform(Xte)
    model=Perceptron(max_iter=1000,random_state=42);model.fit(Xtr_s,ytr);pred=model.predict(Xte_s)
    c=st.columns(4)
    for col,label,val in zip(c,["Accuracy","Precision","Recall","F1-score"],[accuracy_score(yte,pred),precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0)]): col.metric(label,f"{val:.3f}")
    cm=confusion_matrix(yte,pred)
    st.plotly_chart(px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Customers"),x=["Stay","Churn"],y=["Stay","Churn"],title="Confusion matrix"),use_container_width=True)
    st.dataframe(df.head(10),use_container_width=True,hide_index=True)
    st.caption("Metrics are calculated at runtime on a held-out synthetic test set; they are not real customer performance.")

elif page=="Customer Segmentation":
    hero("Customer Segmentation","Custom winner-neuron competitive learning")
    rng=np.random.default_rng(7); n=350
    df=pd.DataFrame({"Monthly spend":rng.gamma(4,25,n),"Visits/month":rng.poisson(8,n),"Orders/month":rng.poisson(3,n)})
    scaler_min=df.min(); scaler_max=df.max(); X=((df-scaler_min)/(scaler_max-scaler_min+1e-9)).to_numpy()
    k=4; weights=np.random.default_rng(11).random((k,X.shape[1]))
    for epoch in range(25):
        for idx in np.random.default_rng(epoch+100).permutation(len(X)):
            x=X[idx]; winner=np.argmin(np.linalg.norm(weights-x,axis=1))
            lr=0.25*(1-epoch/30);weights[winner]+=lr*(x-weights[winner])
    labels=np.array([np.argmin(np.linalg.norm(weights-x,axis=1)) for x in X])
    df["Segment"]=[f"Segment {i+1}" for i in labels]
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(px.scatter(df,x="Monthly spend",y="Visits/month",color="Segment",title="Learned customer groups"),use_container_width=True)
    with c2: st.dataframe(df.groupby("Segment")[["Monthly spend","Visits/month","Orders/month"]].mean().round(2),use_container_width=True)
    st.caption("Custom competitive learning uses a winner neuron and weight updates. It is not K-means.")

elif page=="Algorithm Lab":
    hero("Algorithm Lab","A compact reference for choosing the right technique")
    data=[
      ["BFS","Unweighted graph","Minimum number of edges","Queue"],
      ["DFS","Graph traversal","Explore depth-first; not guaranteed shortest","Stack"],
      ["UCS","Non-negative edge costs","Minimum total path cost","Priority queue"],
      ["A*","Grid / graph + heuristic","Efficient cost-optimal search with admissible heuristic","Priority queue"],
      ["Minimax","Two-player zero-sum game","Best worst-case utility","Recursive game tree"],
      ["Alpha-beta","Minimax game tree","Same decision, potentially fewer nodes","Pruning bounds"],
      ["Perceptron","Binary labelled data","Linear classification","Weight updates"],
      ["Competitive learning","Unlabelled numeric data","Prototype-based grouping","Winner neuron"],
    ]
    st.dataframe(pd.DataFrame(data,columns=["Algorithm","Typical input","Goal","Core mechanism"]),use_container_width=True,hide_index=True)
    st.markdown("#### Route-search comparison")
    g=route_graph(); start=st.selectbox("Compare from",list(g),key="labstart");goal=st.selectbox("Compare to",list(g),index=5,key="labgoal")
    out=[]
    for name,fn in [("BFS",bfs),("DFS",dfs)]:
        p,e=fn(g,start,goal);out.append({"Algorithm":name,"Cost":sum(g[u][v] for u,v in zip(p,p[1:])) if p else None,"Expanded nodes":len(e)})
    p,c,e=ucs(g,start,goal);out.append({"Algorithm":"UCS","Cost":c if p else None,"Expanded nodes":len(e)})
    st.bar_chart(pd.DataFrame(out).set_index("Algorithm")[["Cost","Expanded nodes"]])

else:
    hero("Documentation & Viva","Project guide, limitations, and viva preparation")
    st.markdown("#### System architecture")
    st.code("""User → Streamlit UI → Module logic / algorithms → Visualisations
                         ├── Search algorithms (BFS / DFS / UCS / A*)
                         ├── Rule engine + SQLite (hotel bookings)
                         ├── N-gram text models
                         ├── Neuro-fuzzy / fuzzy controllers
                         ├── Minimax + alpha-beta
                         └── Scikit-learn learning models""")
    st.markdown("#### Practical-to-module mapping")
    mapping=pd.DataFrame([
      ["PEAS / problem formulation","AI Problem Formulation"],["BFS, DFS, UCS","Emergency Route Planner"],
      ["A*","Warehouse Robot"],["PROLOG concepts / rule reasoning","Hotel Knowledge Base (Python fallback; not actual Prolog runtime)"],
      ["Bigram / Trigram language models","Customer Support LLM"],["ANN + fuzzy logic","Traffic Controller"],
      ["Minimax / alpha-beta","Pricing Game"],["Mamdani-style fuzzy controller","Smart Irrigation"],
      ["Mamdani / Sugeno concepts","Credit Risk Fuzzy Systems"],["Perceptron","Customer Churn"],
      ["Competitive learning","Customer Segmentation"]
    ],columns=["Practical topic","Website module"])
    st.dataframe(mapping,use_container_width=True,hide_index=True)
    st.markdown("#### Viva questions")
    qa=[
      ("What is PEAS?","Performance measure, Environment, Actuators and Sensors describe an intelligent agent's task environment."),
      ("When is BFS optimal?","When each edge has equal cost, BFS finds a path with the minimum number of edges."),
      ("How is UCS different from BFS?","UCS expands the lowest cumulative-cost path and handles varying non-negative edge costs."),
      ("What does A* use?","A* ranks a node by f(n)=g(n)+h(n), where g is cost so far and h estimates remaining cost."),
      ("What is a heuristic?","A heuristic estimates the remaining cost from a state to the goal."),
      ("What are facts and rules in Prolog?","Facts store relations; rules derive conclusions from facts and other rules."),
      ("What is unification?","Unification finds substitutions that make two logical terms match."),
      ("What is a bigram?","A bigram is a sequence of two tokens used to estimate the next token."),
      ("What is a trigram?","A trigram is a sequence of three tokens, often conditioned on the previous two tokens."),
      ("What is fuzzy logic?","Fuzzy logic represents degrees of membership instead of only true or false."),
      ("What is defuzzification?","It converts a fuzzy output set into a crisp numerical value."),
      ("What is Minimax?","A game-search method that maximises a player's minimum guaranteed utility."),
      ("Why use alpha-beta pruning?","It skips branches that cannot affect the Minimax decision."),
      ("What is a Perceptron?","A simple linear binary classifier updated using prediction errors."),
      ("What is data leakage?","It occurs when information unavailable at prediction time influences model training or evaluation."),
      ("What is competitive learning?","An unsupervised method where the closest prototype wins and updates toward the input."),
      ("Why normalise features?","Features on different scales can otherwise dominate distance or gradient calculations."),
      ("Why use a test set?","It estimates how well a trained model generalises to unseen examples."),
      ("What is synthetic data?","Artificially generated data used for demonstrations or experiments when real data is unavailable."),
      ("What are model limitations?","Results depend on assumptions, data quality, task complexity, and the gap between simulation and reality.")
    ]
    for q,a in qa:
        with st.expander(q): st.write(a)
    st.markdown("#### Run locally")
    st.code("python -m venv .venv\n# Windows: .venv\\\\Scripts\\\\activate\npip install -r requirements.txt\nstreamlit run app.py",language="bash")
    st.warning("Academic demo only. Hotel booking data is local. Prolog execution, real traffic feeds, hardware control and validated credit decisions are not included.")

