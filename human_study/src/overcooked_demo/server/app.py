import os
import sys
import random
import uuid

# Import and patch the production eventlet server if necessary
if os.getenv("FLASK_ENV", "production") == "production":
    import eventlet

    eventlet.monkey_patch()

import atexit
import json
import logging

# All other imports must come after patch to ensure eventlet compatibility
import pickle
import queue
from datetime import datetime
from threading import Lock

import game
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from game import Game, OvercookedGame, OvercookedTutorial
from utils import ThreadSafeDict, ThreadSafeSet


### Thoughts -- where I'll log potential issues/ideas as they come up
# Should make game driver code more error robust -- if overcooked randomlly errors we should catch it and report it to user
# Right now, if one user 'join's before other user's 'join' finishes, they won't end up in same game
# Could use a monitor on a conditional to block all global ops during calls to _ensure_consistent_state for debugging
# Could cap number of sinlge- and multi-player games separately since the latter has much higher RAM and CPU usage

###########
# Globals #
###########

# Read in global config
CONF_PATH = os.getenv("CONF_PATH", "config.json")
with open(CONF_PATH, "r") as f:
    CONFIG = json.load(f)

# Where errors will be logged
LOGFILE = CONFIG["logfile"]

# Available layout names
LAYOUTS = CONFIG["layouts"]

# Values that are standard across layouts
LAYOUT_GLOBALS = CONFIG["layout_globals"]

# Maximum allowable game length (in seconds)
MAX_GAME_LENGTH = CONFIG["MAX_GAME_LENGTH"]

# Path to where pre-trained agents will be stored on server
AGENT_DIR = CONFIG["AGENT_DIR"]

# Maximum number of games that can run concurrently. Contrained by available memory and CPU
MAX_GAMES = CONFIG["MAX_GAMES"]

# Frames per second cap for serving to client
MAX_FPS = CONFIG["MAX_FPS"]

# Default configuration for predefined experiment
PREDEFINED_CONFIG = json.dumps(CONFIG["predefined"])

# Default configuration for tutorial
TUTORIAL_CONFIG = json.dumps(CONFIG["tutorial"])

# Global queue of available IDs. This is how we synch game creation and keep track of how many games are in memory
FREE_IDS = queue.Queue(maxsize=MAX_GAMES)

# Bitmap that indicates whether ID is currently in use. Game with ID=i is "freed" by setting FREE_MAP[i] = True
FREE_MAP = ThreadSafeDict()

# Initialize our ID tracking data
for i in range(MAX_GAMES):
    FREE_IDS.put(i)
    FREE_MAP[i] = True

# Mapping of game-id to game objects
GAMES = ThreadSafeDict()

# Set of games IDs that are currently being played
ACTIVE_GAMES = ThreadSafeSet()

# Queue of games IDs that are waiting for additional players to join. Note that some of these IDs might
# be stale (i.e. if FREE_MAP[id] = True)
WAITING_GAMES = queue.Queue()

# Mapping of users to locks associated with the ID. Enforces user-level serialization
USERS = ThreadSafeDict()

# Mapping of user id's to the current game (room) they are in
USER_ROOMS = ThreadSafeDict()

GAME_FLOW = ThreadSafeDict()

# Mapping of string game names to corresponding classes
GAME_NAME_TO_CLS = {
    "overcooked": OvercookedGame,
    "tutorial": OvercookedTutorial,
}

game._configure(MAX_GAME_LENGTH, AGENT_DIR)


#######################
# Flask Configuration #
#######################

# Create and configure flask app
app = Flask(__name__, template_folder=os.path.join("static", "templates"))
app.config["DEBUG"] = os.getenv("FLASK_ENV", "production") == "development"
socketio = SocketIO(app, cors_allowed_origins="*", logger=app.config["DEBUG"])


# Attach handler for logging errors to file
handler = logging.FileHandler(LOGFILE)
handler.setLevel(logging.ERROR)
app.logger.addHandler(handler)

app.secret_key = 'abc'

global xai_agent_type, ai_agent_type, isTutorial
xai_agent_type = 'NoX'
ai_agent_type = 'hipt'
user_id = ''
#################################
# Global Coordination Functions #
#################################


def try_create_game(game_name, **kwargs):
    """
    Tries to create a brand new Game object based on parameters in `kwargs`

    Returns (Game, Error) that represent a pointer to a game object, and error that occured
    during creation, if any. In case of error, `Game` returned in None. In case of sucess,
    `Error` returned is None

    Possible Errors:
        - Runtime error if server is at max game capacity
        - Propogate any error that occured in game __init__ function
    """
    try:
        curr_id = FREE_IDS.get(block=False)
        assert FREE_MAP[curr_id], "Current id is already in use"
        game_cls = GAME_NAME_TO_CLS.get(game_name, OvercookedGame)
        game = game_cls(id=curr_id, **kwargs)
    except queue.Empty:
        err = RuntimeError("Server at max capacity")
        return None, err
    except Exception as e:
        return None, e
    else:
        GAMES[game.id] = game
        FREE_MAP[game.id] = False
        return game, None


def cleanup_game(game: OvercookedGame):
    if FREE_MAP[game.id]:
        raise ValueError("Double free on a game")

    # User tracking
    for user_id in game.players:
        leave_curr_room(user_id)

    # Socketio tracking
    socketio.close_room(game.id)
    # Game tracking
    FREE_MAP[game.id] = True
    FREE_IDS.put(game.id)
    del GAMES[game.id]

    if game.id in ACTIVE_GAMES:
        ACTIVE_GAMES.remove(game.id)


def get_game(game_id):
    return GAMES.get(game_id, None)


def get_curr_game(user_id):
    return get_game(get_curr_room(user_id))


def get_curr_room(user_id):
    return USER_ROOMS.get(user_id, None)


def set_curr_room(user_id, room_id):
    USER_ROOMS[user_id] = room_id


def leave_curr_room(user_id):
    del USER_ROOMS[user_id]


def get_waiting_game():
    """
    Return a pointer to a waiting game, if one exists

    Note: The use of a queue ensures that no two threads will ever receive the same pointer, unless
    the waiting game's ID is re-added to the WAITING_GAMES queue
    """
    try:
        waiting_id = WAITING_GAMES.get(block=False)
        while FREE_MAP[waiting_id]:
            waiting_id = WAITING_GAMES.get(block=False)
    except queue.Empty:
        return None
    else:
        return get_game(waiting_id)


##########################
# Socket Handler Helpers #
##########################


def _leave_game(user_id):
    """
    Removes `user_id` from it's current game, if it exists. Rebroadcast updated game state to all
    other users in the relevant game.

    Leaving an active game force-ends the game for all other users, if they exist

    Leaving a waiting game causes the garbage collection of game memory, if no other users are in the
    game after `user_id` is removed
    """
    # Get pointer to current game if it exists
    print("__Leaving game__")
    game = get_curr_game(user_id)

    if not game:
        # Cannot leave a game if not currently in one
        return False

    # Acquire this game's lock to ensure all global state updates are atomic
    with game.lock:
        # Update socket state maintained by socketio
        leave_room(game.id)

        # Update user data maintained by this app
        leave_curr_room(user_id)

        # Update game state maintained by game object
        if user_id in game.players:
            game.remove_player(user_id)
        else:
            game.remove_spectator(user_id)

        # Whether the game was active before the user left
        was_active = game.id in ACTIVE_GAMES

        # Rebroadcast data and handle cleanup based on the transition caused by leaving
        if was_active and game.is_empty():
            # Active -> Empty
            game.deactivate()
        elif game.is_empty():
            # Waiting -> Empty
            cleanup_game(game)
        elif not was_active:
            # Waiting -> Waiting
            emit("waiting", {"in_game": True}, room=game.id)
        elif was_active and game.is_ready():
            # Active -> Active
            pass
        elif was_active and not game.is_empty():
            # Active -> Waiting
            game.deactivate()

    return was_active


def _create_game(user_id,
                 game_name,
                 params={},
                 **kwargs):
    current_phase=kwargs.get("current_phase",1)
    current_session=kwargs.get("current_session",1)
    current_round=kwargs.get("current_round",1)
    layouts=kwargs.get("layouts",[]) or params.get("layouts", [])
    layouts_order=kwargs.get("layouts_order",[])
    game_flow_on = kwargs.get('game_flow_on', 0)
    is_ending = kwargs.get('is_ending', 0)
    xai_agent_assignment = kwargs.get('xai_agent_assignment', [])
    ai_agent_assignment = kwargs.get('ai_agent_assignment', [])
    params.update({
        "current_session": current_session,
        "current_phase" : current_phase,
        "current_round": current_round,
        "total_rounds": CONFIG["total_num_rounds"],
        "layouts": layouts
    })
    game, err = try_create_game(game_name, **params)
    if not game:
        emit("creation_failed", {"error": err.__repr__()})
        return
    spectating = True
    with game.lock:
        if not game.is_full():
            spectating = False
            game.add_player(user_id)
        else:
            spectating = True
            game.add_spectator(user_id)
        join_room(game.id)
        set_curr_room(user_id, game.id)
        if game.is_ready():
            game.activate()
            game.update_explanation('')
            ACTIVE_GAMES.add(game.id)
            start_info = game.to_json()
            start_info["currentSession"] = current_session
            start_info["currentRound"] = current_round
            start_info["totalRounds"] = CONFIG["total_num_rounds"]
            
            # Transform each element: replace underscores with spaces, then title-case
            display_order = [layout.replace("_", " ").title() for layout in layouts_order]

            start_info["experiment_order_disp"] = " => ".join(display_order)
        
            if ai_agent_assignment:
                start_info["aiAgentType"] = ai_agent_assignment[current_phase-1]
            else:
                start_info["aiAgentType"] = params.get("aiAgentType", ai_agent_type)
            
            if xai_agent_assignment:
                start_info["xaiAgentType"] = xai_agent_assignment[current_phase-1]
            else:
                start_info["xaiAgentType"] = params.get("xaiAgentType", xai_agent_type)
            start_info["current_layout"] = game.curr_layout
            print(f"Current Phase: {current_phase} & Current session: {current_session} & Current round: {current_round}\n")
            print("[XAI] Agent type: ", start_info["xaiAgentType"])
            start_info["disable_xai"] = CONFIG["disable_xai"]
            emit(
                "start_game",
                {"spectating": spectating, "start_info": start_info},
                room=game.id,
            )
            emit(
                "start_sensors",
                {"spectating": spectating, "start_info": {"round_id": game.id, "player_id": user_id, "uid": session["user_id"], "xaiAgentType": start_info["xaiAgentType"]}},
                broadcast=True
            )
            #game.start_recording_kb_events(f"{game.id}_{session['user_id']}")
            socketio.start_background_task(play_game, game, fps=6, game_flow_on=game_flow_on, is_ending=is_ending)
        else:
            WAITING_GAMES.put(game.id)
            emit("waiting", {"in_game": True}, room=game.id)


#####################
# Debugging Helpers #
#####################


def _ensure_consistent_state():
    """
    Simple sanity checks of invariants on global state data

    Let ACTIVE be the set of all active game IDs, GAMES be the set of all existing
    game IDs, and WAITING be the set of all waiting (non-stale) game IDs. Note that
    a game could be in the WAITING_GAMES queue but no longer exist (indicated by
    the FREE_MAP)

    - Intersection of WAITING and ACTIVE games must be empty set
    - Union of WAITING and ACTIVE must be equal to GAMES
    - id \in FREE_IDS => FREE_MAP[id]
    - id \in ACTIVE_GAMES => Game in active state
    - id \in WAITING_GAMES => Game in inactive state
    """
    waiting_games = set()
    active_games = set()
    all_games = set(GAMES)

    for game_id in list(FREE_IDS.queue):
        assert FREE_MAP[game_id], "Freemap in inconsistent state"

    for game_id in list(WAITING_GAMES.queue):
        if not FREE_MAP[game_id]:
            waiting_games.add(game_id)

    for game_id in ACTIVE_GAMES:
        active_games.add(game_id)

    assert (
        waiting_games.union(active_games) == all_games
    ), "WAITING union ACTIVE != ALL"

    assert not waiting_games.intersection(
        active_games
    ), "WAITING intersect ACTIVE != EMPTY"

    assert all(
        [get_game(g_id)._is_active for g_id in active_games]
    ), "Active ID in waiting state"
    assert all(
        [not get_game(g_id)._id_active for g_id in waiting_games]
    ), "Waiting ID in active state"


def get_agent_names():
    return [
        d
        for d in os.listdir(AGENT_DIR)
        if os.path.isdir(os.path.join(AGENT_DIR, d))
    ]


######################
# Application routes #
######################

# Hitting each of these endpoints creates a brand new socket that is closed
# at after the server response is received. Standard HTTP protocol


@app.route("/", methods=["GET", "POST"])
def index():
    agent_names = get_agent_names()
    # Check if the form was submitted (POST request)
    show_modal = False
    if request.method == "POST":
        # Get the UID from the form
        uid = request.form.get('uid')
        print(f"Received UID: {uid}")

        # Store the UID in the session (Flask's session management)
        session['user_id'] = uid
        # Optionally, store the UID in the GameSession class
        OvercookedGame.set_uid(uid)
        show_modal = True  # signal to show modal after post
    else:  # GET request
        # Try to get UID from session first
        uid = session.get('user_id')

        # If session is empty, try MTurk URL parameter
        if not uid:
            uid = request.args.get('workerId')

        # If still empty (e.g., local testing), set a default
        if not uid:
            #uid = "LOCAL_TEST_USER"
            uid = str(uuid.uuid4())

        # Save it back to session so it exists for SocketIO
        session['user_id'] = uid
        OvercookedGame.set_uid(uid)

    # Randomize default layout loading
    default_layouts = CONFIG["layouts"].copy()
    random.shuffle(default_layouts)
    default_layout = default_layouts[0] if CONFIG["randomize_layout"] else CONFIG["default_layout"]

    return render_template(
        "index.html",
        agent_names=agent_names, 
        layouts=LAYOUTS,
        uid = uid,
        show_modal=show_modal,  # <== pass to frontend
        default_agent=CONFIG["layout_agent_mapping"][default_layout],
        default_layout=default_layout,
        enable_survey=CONFIG['enable_survey'],
        disable_close=CONFIG['disable_close'],
        disable_xai=CONFIG['disable_xai']
    )

@app.route("/get_config", methods=["GET"])
def get_config():
    resp = {"config_data":CONFIG}
    return jsonify(resp)

from flask import jsonify

@app.route("/reset_uid", methods=["POST"])
def reset_uid():
    session.pop('user_id', None)
    return jsonify({"status": "success", "message": "User ID reset."})

# @app.route('/set_user', methods=['POST'])
# def set_user():
#     # Get the UID from the form
#     uid = request.form.get('uid')

#     # Store the UID in the session (Flask's session management)
#     session['user_id'] = uid

#     # Optionally, store the UID in the GameSession class
#     OvercookedGame.set_uid(uid)

#     # Redirect to the game page or another appropriate route
#     #return render_template("index.html", uid=uid)

@app.route("/predefined")
def predefined():
    uid = request.args.get("UID")
    num_layouts = len(CONFIG["predefined"]["experimentParams"]["layouts"])

    return render_template(
        "predefined.html",
        uid=uid,
        config=PREDEFINED_CONFIG,
        num_layouts=num_layouts,
    )


@app.route("/instructions")
def instructions():
    return render_template("instructions.html", layout_conf=LAYOUT_GLOBALS)

import time
@app.route("/tutorial")
def tutorial():
    time.sleep(0.5)
    print("TUTORIAL_CONFIG ", TUTORIAL_CONFIG)
    return render_template("tutorial.html", 
                           config=TUTORIAL_CONFIG, 
                           enable_survey=CONFIG['enable_survey'],
                           disable_close=CONFIG['disable_close']
                           )


@app.route("/debug")
def debug():
    resp = {}
    games = []
    active_games = []
    waiting_games = []
    users = []
    free_ids = []
    free_map = {}
    for game_id in ACTIVE_GAMES:
        game = get_game(game_id)
        active_games.append({"id": game_id, "state": game.to_json()})

    for game_id in list(WAITING_GAMES.queue):
        game = get_game(game_id)
        game_state = None if FREE_MAP[game_id] else game.to_json()
        waiting_games.append({"id": game_id, "state": game_state})

    for game_id in GAMES:
        games.append(game_id)

    for user_id in USER_ROOMS:
        users.append({user_id: get_curr_room(user_id)})

    for game_id in list(FREE_IDS.queue):
        free_ids.append(game_id)

    for game_id in FREE_MAP:
        free_map[game_id] = FREE_MAP[game_id]

    resp["active_games"] = active_games
    resp["waiting_games"] = waiting_games
    resp["all_games"] = games
    resp["users"] = users
    resp["free_ids"] = free_ids
    resp["free_map"] = free_map
    return jsonify(resp)


#########################
# Socket Event Handlers #
#########################

# Asynchronous handling of client-side socket events. Note that the socket persists even after the
# event has been handled. This allows for more rapid data communication, as a handshake only has to
# happen once at the beginning. Thus, socket events are used for all game updates, where more rapid
# communication is needed


def creation_params(params):
    """
    This function extracts the dataCollection and oldDynamics settings from the input and
    process them before sending them to game creation
    """
    # this params file should be a dictionary that can have these keys:
    # playerZero: human/Rllib*agent
    # playerOne: human/Rllib*agent
    # layout: one of the layouts in the config file, I don't think this one is used
    # gameTime: time in seconds
    # oldDynamics: on/off
    # dataCollection: on/off
    # layouts: [layout in the config file], this one determines which layout to use, and if there is more than one layout, a series of game is run back to back
    #
    use_old = False
    global xai_agent_type, ai_agent_type, isTutorial
    if "oldDynamics" in params and params["oldDynamics"] == "on":
        params["mdp_params"] = {"old_dynamics": True}
        use_old = True  
    if "dataCollection" in params and params["dataCollection"] == "on":
        # config the necessary setting to properly save data
        params["dataCollection"] = True
        mapping = {"human": "H"}
        # gameType is either HH, HA, AH, AA depending on the config
        gameType = "{}{}".format(
            mapping.get(params["playerZero"], "A"),
            mapping.get(params["playerOne"], "A"),
        )
        params["collection_config"] = {
            "time": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
            "type": gameType,
        }
        if use_old:
            params["collection_config"]["old_dynamics"] = "Old"
        else:
            params["collection_config"]["old_dynamics"] = "New"
    else:
        params["dataCollection"] = False
    
    if "xaiAgentType" in params:
        xai_agent_type = params["xaiAgentType"]
    if "aiAgentType" in params:
        ai_agent_type = params["aiAgentType"]
    isTutorial = True if "isTutorial" in params else False 

@socketio.on("create-next")
def on_create_next(data):
    global user_id

    user_id = request.sid or user_id

    with USERS[user_id]:
        # Retrieve current game if one exists
        curr_game = get_curr_game(user_id)
        if curr_game:
            # Cannot create if currently in a game
            return

        params = data.get("params", {})
        creation_params(params)

        game_name = data.get("game_name", "overcooked")
        # all_layouts = CONFIG["layouts"].copy()
        process_game_flow(curr_game, params)

        layouts = GAME_FLOW['all_layouts']


        params["layouts"] = layouts
        if GAME_FLOW:
            # params["playerOne"] = GAME_FLOW["playerOne"]
            params["playerZero"] = GAME_FLOW["playerZero"]
            params["layout"] = layouts[GAME_FLOW["current_session"]-1]

            _create_game(user_id=user_id,
                        game_name=game_name,
                        params=params,
                        current_phase = GAME_FLOW['current_phase'],
                        current_round=GAME_FLOW["current_round"],
                        current_session=GAME_FLOW["current_session"],
                        layouts=[layouts[GAME_FLOW["current_session"]-1]],
                        layouts_order=GAME_FLOW['all_layouts'],
                        game_flow_on=1,
                        is_ending=GAME_FLOW["is_ending"],
                        xai_agent_assignment=GAME_FLOW['xai_agent_assignment'])


def process_game_flow(curr_game, params):
    current_phase = GAME_FLOW['current_phase']
    current_round = GAME_FLOW['current_round']
    current_session = GAME_FLOW['current_session']
    total_rounds = GAME_FLOW['total_num_rounds']
    
    layout_agent_mapping = CONFIG["layout_agent_mapping"]
    if current_round < total_rounds:
        GAME_FLOW['current_round'] = current_round + 1
    elif current_round >= total_rounds:
        print("Moving to new session...")

        if current_session < len(GAME_FLOW['all_layouts']):
            # Resetting to initial round 1 with new session and new layout
            GAME_FLOW['current_round'] = 1
            GAME_FLOW['current_session'] = current_session + 1
    if current_session > len(GAME_FLOW['all_layouts'])-1 and current_phase < GAME_FLOW['total_phases']:
        print("Moving to new Phase...")
        GAME_FLOW['current_phase'] = current_phase + 1
        GAME_FLOW['current_round'] = CONFIG['initial_round']
        GAME_FLOW['current_session'] = CONFIG['initial_session']

        all_layouts = GAME_FLOW['all_layouts'].copy()
        layout_agent_mapping = CONFIG["layout_agent_mapping_hrl"]
        
        #Shuffle game layout order
        # all_layouts.remove(params["layout"])
        random.shuffle(all_layouts)
        layouts = all_layouts
        if layouts ==  GAME_FLOW['all_layouts']:
            print("Same layout order. Shuffling again")
            all_layouts = GAME_FLOW['all_layouts'].copy()
            #Shuffle game layout order
            # all_layouts.remove(params["layout"])
            random.shuffle(all_layouts)
            layouts = all_layouts
        print(f"Phase {GAME_FLOW['current_phase'] } - New Layout Order {layouts}")
        GAME_FLOW['all_layouts'] =  layouts

    if GAME_FLOW['current_phase'] >= GAME_FLOW['total_phases'] and GAME_FLOW['current_session'] >= len(GAME_FLOW['all_layouts']) and GAME_FLOW['current_round'] >= GAME_FLOW['total_num_rounds']:
        GAME_FLOW['is_ending'] = 1
    
    # GAME_FLOW["playerOne"] = layout_agent_mapping[GAME_FLOW['all_layouts'][GAME_FLOW['current_session']-1]]
    GAME_FLOW["playerZero"] = layout_agent_mapping[GAME_FLOW['all_layouts'][GAME_FLOW['current_session']-1]]
    params['ai_agent_assignment'] = CONFIG["ai_assignment"]
     
@socketio.on("create")
def on_create(data):
    global user_id

    user_id = request.sid or user_id

    with USERS[user_id]:
        # Retrieve current game if one exists
        curr_game = get_curr_game(user_id)
        if curr_game:
            # Cannot create if currently in a game
            return

        params = data.get("params", {})

        creation_params(params)

        game_name = data.get("game_name", "overcooked")
        all_layouts = CONFIG["layouts"].copy()
        #Shuffle game layout order
        all_layouts.remove(params["layout"])
        random.shuffle(all_layouts)
        layouts = [params["layout"]] + all_layouts
        print("New Layout Order ", layouts)
        disable_xai = CONFIG["disable_xai"]
        # Retrieve randomized XAI agent order
        xai_agent_assignment = CONFIG["xai_assignment"]
        ai_agent_assignment = CONFIG["ai_assignment"]
        # if CONFIG["randomize_xai"]:
        #     xai_agent_assignment = assignXAIAgents(user_id)
        print("XAI Agent order: ", xai_agent_assignment)

        params["layouts"] = layouts
        print("Create game params: ",params)
        GAME_FLOW['current_session'] =  CONFIG["initial_session"]
        GAME_FLOW['current_phase'] = CONFIG['initial_phase']
        GAME_FLOW['current_round'] =  CONFIG["initial_round"]
        GAME_FLOW['total_num_rounds'] =  CONFIG["total_num_rounds"]
        GAME_FLOW['total_phases'] = 3 if isTutorial else len(ai_agent_assignment) #len(xai_agent_assignment)
        GAME_FLOW['all_layouts'] =  layouts
        # GAME_FLOW['prev_params'] = layouts
        GAME_FLOW['is_ending'] = CONFIG["is_ending"]
        GAME_FLOW['xai_agent_assignment'] = xai_agent_assignment
        GAME_FLOW['ai_agent_assignment'] = ai_agent_assignment
        params['ai_agent_assignment'] = ai_agent_assignment
        # initialise => @TODO check
        GAME_FLOW["playerZero"] = 'human'
        GAME_FLOW["playerOne"] = 'human'
        print("params:  ", params)
        if params.get("playerZero") != "human":
            GAME_FLOW["playerZero"] = params.get("playerZero")

        if params.get("playerOne") != "human":
            GAME_FLOW["playerOne"] = params.get("playerOne")
        
        # params["playerOne"] = GAME_FLOW["playerOne"]
        params["playerZero"] = GAME_FLOW["playerZero"]
        # params["playerOne"] = GAME_FLOW["playerOne"]
        # if "playerOne" in GAME_FLOW:
        #     params["playerOne"] = GAME_FLOW["playerOne"]
        # else:
        #     # Handle the missing key appropriately
        #     print("WARNING: 'playerOne' not found in GAME_FLOW.")
        #     params["playerOne"] = "human"  # or whatever fallback makes sense

        _create_game(user_id=user_id,
                    game_name=game_name,
                    params=params,
                    current_session=CONFIG["initial_session"],
                    current_phase=CONFIG["initial_phase"],
                    current_round=CONFIG["initial_round"],
                    layouts=[layouts[CONFIG["initial_session"]-1]],
                    layouts_order=layouts, 
                    game_flow_on=CONFIG['game_flow_on'],
                    xai_agent_assignment=xai_agent_assignment,
                    ai_agent_assignment=ai_agent_assignment)


@socketio.on("join")
def on_join(data):
    user_id = request.sid
    with USERS[user_id]:
        create_if_not_found = data.get("create_if_not_found", True)

        # Retrieve current game if one exists
        curr_game = get_curr_game(user_id)
        if curr_game:
            # Cannot join if currently in a game
            return

        # Retrieve a currently open game if one exists
        game = get_waiting_game()

        if not game and create_if_not_found:
            # No available game was found so create a game
            params = data.get("params", {})
            creation_params(params)
            params['ai_agent_assignment'] = CONFIG["ai_assignment"]
            game_name = data.get("game_name", "overcooked")
            _create_game(user_id=user_id,
                         game_name=game_name,
                         params=params)
            return

        elif not game:
            # No available game was found so start waiting to join one
            emit("waiting", {"in_game": False})
        else:
            # Game was found so join it
            with game.lock:
                join_room(game.id)
                set_curr_room(user_id, game.id)
                game.add_player(user_id)

                if game.is_ready():
                    # Game is ready to begin play
                    game.activate()
                    game.update_explanation('')
                    start_info = game.to_json()
                    # start_info["xaiAgentType"] = params["xaiAgentType"] #May be not need in tutorial
                    ACTIVE_GAMES.add(game.id)
                    emit(
                        "start_game",
                        {"spectating": False, "start_info": game.to_json()},
                        room=game.id,
                    )
                    emit(
                        "start_sensors",
                        {"spectating": False, "start_info":  {"round_id": game.id, "player_id": user_id, "uid": session["user_id"], "xaiAgentType": params["xaiAgentType"]}},
                        broadcast=True
                    )
                    socketio.start_background_task(play_game, game)
                else:
                    # Still need to keep waiting for players
                    WAITING_GAMES.put(game.id)
                    emit("waiting", {"in_game": True}, room=game.id)


@socketio.on("leave")
def on_leave(data):
    user_id = request.sid
    print("Ending game data: ",data)
    with USERS[user_id]:
        was_active = _leave_game(user_id)

        if was_active:
            emit("end_game", {"status": Game.Status.DONE, "data": {}})
            emit("stop_sensors", {"status": Game.Status.DONE, "data": {}}, broadcast=True)
        else:
            emit("end_lobby")


@socketio.on("action")
def on_action(data):
    user_id = request.sid
    action = data["action"]

    game = get_curr_game(user_id)
    if not game:
        return

    game.enqueue_action(user_id, action)


@socketio.on("connect")
def on_connect():
    user_id = request.sid

    if user_id in USERS:
        return

    USERS[user_id] = Lock()

# TODO: remove adax UI element if adax is unchecked
@socketio.on("xai_message")
def on_xai_message(data):
    user_id = request.sid
    adaxplanation = data["explanation"]
    game = next(iter(GAMES.values()))
    if not game:
        return
    if xai_agent_type in ["StaticX","AdaX"]:
        socketio.emit("xai_voice",
             {"explanation": adaxplanation})
        game.update_explanation(adaxplanation)
        print("adaxplanation: ", adaxplanation)

@socketio.on("disconnect")
def on_disconnect():
    print("disconnect triggered", file=sys.stderr)
    # Ensure game data is properly cleaned-up in case of unexpected disconnect
    user_id = request.sid
    if user_id not in USERS:
        return
    with USERS[user_id]:
        _leave_game(user_id)

    del USERS[user_id]


# Exit handler for server
def on_exit():
    # Force-terminate all games on server termination
    print("Exiting...")
    for game_id in GAMES:
        socketio.emit(
            "end_game",
            {
                "status": Game.Status.INACTIVE,
                "data": get_game(game_id).get_data(),
            },
            room=game_id,
        )
        socketio.emit(
            "stop_sensors",
            {
                "status": Game.Status.INACTIVE,
                "data": get_game(game_id).get_data(),
            },
            broadcast=True
        )
        


#############
# Game Loop #
#############


def play_game(game: OvercookedGame, fps=6, game_flow_on=0, is_ending=0):
    """
    Asynchronously apply real-time game updates and broadcast state to all clients currently active
    in the game. Note that this loop must be initiated by a parallel thread for each active game

    game (Game object):     Stores relevant game state. Note that the game id is the same as to socketio
                            room id for all clients connected to this game
    fps (int):              Number of game ticks that should happen every second
    """
    try:
        status = Game.Status.ACTIVE
        while status != Game.Status.DONE and status != Game.Status.INACTIVE:
            with game.lock:
                status = game.tick()
            if status == Game.Status.RESET:
                with game.lock:
                    data = game.get_data()
                socketio.emit(
                    "reset_game",
                    {
                        "state": game.to_json(),
                        "timeout": game.reset_timeout,
                        "data": data,
                    },
                    room=game.id,
                )
                socketio.sleep(game.reset_timeout / 1000)
            else:
                socketio.emit(
                    "state_pong", {"state": game.get_state()}, room=game.id
                )
            socketio.sleep(1 / fps)

        with game.lock:
            data = game.get_data()
            
            data['game_flow_on'] = 0 if is_ending  else game_flow_on 
            data['is_ending'] = is_ending

            data['session_id'] = GAME_FLOW['current_session'] if GAME_FLOW else ''
            tut_config = json.loads(TUTORIAL_CONFIG)
            data['layout'] = GAME_FLOW['all_layouts'][GAME_FLOW['current_session']-1] if GAME_FLOW else tut_config['tutorialParams']['layouts'][0]
            data['xai_agent'] = GAME_FLOW['xai_agent_assignment'][GAME_FLOW['current_phase']-1] if GAME_FLOW else ''
            data['ai_agent'] = GAME_FLOW['ai_agent_assignment'][GAME_FLOW['current_phase']-1] if GAME_FLOW else ''
            data['session_ended'] = False
            data['phase_ended'] = False
            data['game_ended'] = False
            data['survey_baseurl'] = None
            # # check session end status => DISABLED FOR HRL
            # if(GAME_FLOW and GAME_FLOW['current_round'] == GAME_FLOW['total_num_rounds']):
            #     data['session_ended'] = True
            #     # data['survey_baseurl'] = CONFIG['questionnaire_links']['post_session'] if data['xai_agent'] != 'NoX' else CONFIG['questionnaire_links']['post_session_nox'] # add differnt url for NoX
            #     data['survey_baseurl'] = CONFIG['questionnaire_links']['post_session']
            
            # check session end status
            if(GAME_FLOW and GAME_FLOW['current_session'] >= len(GAME_FLOW['all_layouts'])):
                data['phase_ended'] = True
                data['survey_baseurl'] = CONFIG['questionnaire_links']['post_session'] #ADDED DIFF URL FOR HRL
            
            # check game end status
            if GAME_FLOW and GAME_FLOW['current_phase'] >= GAME_FLOW['total_phases'] and GAME_FLOW['current_session'] >= len(GAME_FLOW['all_layouts']) and GAME_FLOW['current_round'] >= GAME_FLOW['total_num_rounds']:
                data['game_ended'] = True  
                data['survey_baseurl'] = CONFIG['questionnaire_links']['post_session'] #ADDED DIFF URL FOR HRL
                data['survey_baseurl_end'] = CONFIG['questionnaire_links']['post_game'] # DISABLED FOR HRL

            socketio.emit(
                "end_game", {"status": status, "data": data}, room=game.id
            )
            socketio.emit(
                "stop_sensors", {"status": status, "data": data}, broadcast=True
            )
            #game.stop_recording_kb_events()

            if status != Game.Status.INACTIVE:
                game.deactivate()
            cleanup_game(game)
    except Exception as e:
        print("game play error ", e)

if __name__ == "__main__":
    # Dynamically parse host and port from environment variables (set by docker build)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    print("socket ", host, port)
    # Attach exit handler to ensure graceful shutdown
    atexit.register(on_exit)

    # https://localhost:80 is external facing address regardless of build environment
    socketio.run(app, host=host, port=port, log_output=app.config["DEBUG"], debug=True)
