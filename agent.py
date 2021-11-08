from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import Unit
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math, sys
import numpy as np
import random
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import math

from pathlib import Path
p = Path('/kaggle_simulations/agent/')
if p.exists():
    sys.path.append(str(p))
else:
    p = Path('__file__').resolve().parent


game_state = None

model = None

def get_inputs(game_state):
    # Teh shape of the map
    w,h = game_state.map.width, game_state.map.height
    # The map of ressources
    M = [ [0  if game_state.map.map[j][i].resource==None else game_state.map.map[j][i].resource.amount for i in range(w)]  for j in range(h)]
    
    M = np.array(M).reshape((h,w,1))
    
    # The map of units features
    U_player = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]    
    units = game_state.players[0].units
    for i in units:
        U_player[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]
    U_player = np.array(U_player)
    
    U_opponent = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]
    units = game_state.players[1].units
    for i in units:
        U_opponent[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]

    U_opponent = np.array(U_opponent)
    
    # The map of cities featrues
    e = game_state.players[0].cities
    C_player = [ [[0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_player[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep]
    C_player = np.array(C_player)

    e = game_state.players[1].cities
    C_opponent = [ [[0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_opponent[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep]
    C_opponent = np.array(C_opponent)
    
    # stacking all in one array
    E = np.dstack([M,U_opponent,U_player,C_opponent,C_player])
    return E


def get_direction(action):
    return "csnwe"[action] if action < 5 else None


def get_unit_future_position(unit, direction):
    to_x = unit.pos.x
    to_y = unit.pos.y

    if direction == "e":
        to_x += 1
    elif direction == "s":
        to_y += 1
    elif direction == "w":
        to_x -= 1
    elif direction == "n":
        to_y -= 1
        
    return to_x, to_y


def is_unit_action_valid(unit, option, actions, player, opponent):
    height, width = game_state.map.width, game_state.map.height
    
    if not unit.can_act():
        return False
    
    # if option == move:
    if option < 5:
        to_x, to_y = get_unit_future_position(unit, get_direction(option))

        # Out of bond
        if to_x < 0 or to_x >= width or to_y < 0 or to_y >= height:
            return False

        to_cell = game_state.map.get_cell(to_x, to_y)
        to_citytile = to_cell.citytile

        # Not citytile and cell already has unit
        if to_citytile is None:
            has_player_unit = to_cell.has_player_unit(player)
            has_opponent_unit = to_cell.has_player_unit(opponent)
            
            if has_player_unit or has_opponent_unit:
                return False

            for action in actions:
                # Move action string is "m {} {}".format(self.id, dir)
                action = action.split(" ")

                if action[0] != "m":
                    continue

                _, player_unit_id, direction = action

                player_unit_to_x = player_unit_to_y = None
                for player_unit in player.units:
                    if player_unit.id != player_unit_id:
                        continue

                    player_unit_to_x, player_unit_to_y = get_unit_future_position(player_unit, direction)
                    break

                if player_unit_to_x is None or player_unit_to_y is None:
                    continue

                if to_x == player_unit_to_x and to_y == player_unit_to_y:
                    return False
                    
        # Opponent citytile
        elif to_citytile.team == opponent.team:
            return False
    #elif option == build_city:
    elif option == 5:
        if not unit.can_build(game_state.map):
            return False
    else: return False
    '''elif option == pillage:
        to_cell = get_cell(to_x, to_y)

        # Not road
        if to_cell.road == 0:
            return False'''

    return True


def is_city_tile_action_valid(city_tile, action, player):
    if not city_tile.can_act():
        return False
    
    #if action == research:
    if action == 6:
        pass
    #elif action == build_worker or action == build_cart:
    elif action == 7:
        owned_units = len(player.units)
        owned_city_tiles = 0
        
        for city in player.cities.values():
            owned_city_tiles += len(city.citytiles)

        if owned_units >= owned_city_tiles:
            return False
    else: return False
        
    return True


def get_best_unit_valid_action(unit, options, actions, player, opponent, i=1):
    if i == len(options):
        return -1
    
    option = np.argsort(options)[-i]
    
    if is_unit_action_valid(unit, option, actions, player, opponent):
        return option
    
    return get_best_unit_valid_action(unit, options, actions, player, opponent, i + 1)


def get_best_city_tile_valid_action(city_tile, options, player, i=1):
    if i == len(options):
        return -1
    
    option = np.argsort(options)[-i]
    
    if is_city_tile_action_valid(city_tile, option, player):
        return option
    
    return get_best_city_tile_valid_action(city_tile, options, player, i + 1)


def get_model(s):
    input_shape = (s,s,17)
    inputs = keras.Input(shape= input_shape,name = 'TheGameMap')
    f = layers.Flatten()(inputs)   
    h,w,_ = get_inputs(game_state).shape
    print(h,w)
#     output = layers.Dense(w*h*8,activation = "sigmoid")(f)
    
    f = layers.Dense(w*h,activation = "sigmoid")(f)
    f = layers.Reshape((h,w,-1))(f)
    units = layers.Dense(6,activation = "softmax",name = "Units_actions")(f)
    
    cities = layers.Dense(2,activation = "sigmoid",name = "Cities_actions")(f)
    
    output = layers.Concatenate()([units,cities])
    model = keras.Model(inputs = inputs, outputs = output)
    return model


def get_prediction_actions(y_A, y_B, player, opponent):
    actions = []
    best_options = np.zeros((game_state.map.width, game_state.map.height), dtype=int)

    for unit in player.units:
        unit_y, unit_x = unit.pos.y, unit.pos.x

        options = y_A[unit_y][unit_x] + y_B[unit_y][unit_x]
        
        best_option = get_best_unit_valid_action(unit, options, actions, player, opponent)
        best_options[unit_y, unit_x] = best_option

        if -1 < best_option < 5:
            actions.append(unit.move(get_direction(best_option)))
        elif best_option == 5:
            actions.append(unit.build_city())
            
    for city in player.cities.values():
        for city_tile in city.citytiles:
            city_tile_y, city_tile_x = city_tile.pos.y, city_tile.pos.x
            
            options = y_A[city_tile_y][city_tile_x] + y_B[city_tile_y][city_tile_x]
            
            best_option = get_best_city_tile_valid_action(city_tile, options, player)
            best_options[city_tile_y, city_tile_x] = best_option
        
            if best_option == 6:
                actions.append(city_tile.research())
            elif best_option == 7:
                actions.append(city_tile.build_worker())
    
    return actions, best_options

FILE_NAME_A = "model_A_12_10.h5"
FILE_NAME_B = "model_B_12_10.h5"

def agent(observation, configuration):
    global game_state, epsilon, model_A, model_B
    
    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        print("Creating model..")
        model_A = get_model(game_state.map.width)
        model_B = get_model(game_state.map.width)
        print("Load model weight..")
        try:
            model_A.load_weights(f'model_A_{game_state.map.width}.h5', by_name=True, skip_mismatch=True)
            model_B.load_weights(f'model_B_{game_state.map.width}.h5', by_name=True, skip_mismatch=True)
        except Exception as e:
            print('Error in model load')
            print(e)
#         model = tf.keras.models.load_model('model.h5')
        print("Done crating mdoel")
        
        
    else:
        game_state._update(observation["updates"])
    

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Get Prediction of actions
    x = get_inputs(game_state)
    y_A = model_A.predict(np.asarray([x]))[0]
    y_B = model_B.predict(np.asarray([x]))[0]
    
    print(f"Step: {observation['step']}")
    
    actions, _ = get_prediction_actions(y_A, y_B, player, opponent)
    
    return actions
