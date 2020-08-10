extends Node2D

#Global variables:
	#default direction:
var direction = Vector2(1, 0)
	#variables for instances:
		#snake;s body
onready var body = preload("res://Scenes/Body.tscn")
		#snake's tail
onready var tail = preload("res://Scenes/Tail.tscn")
	#gap is a constant delta between two body blocks
const gap = -6
	#variables for move_snake(): 
		#direction of next body block
var next_tail_dir = Vector2(1, 0)
		#direction of previous body block
var prev_tail_dir = Vector2(1, 0)
	#variables to Game.gd:
		#start position is position, where snake starts moving
var default_position = Vector2.ZERO
		#queue of inputs: each keyboard input(up, down, right, left) is here
var dir_queue = []

func _ready():
	#default position is start position
	default_position = $head.position
	#default snake is snake with tail element
	add_tail()

func _physics_process(delta):
	#local variable; push_v stores the direction corresponding to the last input
	#At the end of this function push_v is the last element in dir_queue
	var push_v
	if(Input.is_action_pressed("ui_down")):
		push_v = Vector2(0, 1)
		
		#if the last element in dir_queue is equal to push_v, it won't be added
		if(dir_queue.back() != push_v):
			dir_queue.push_back(push_v)
	elif(Input.is_action_pressed("ui_up")):
		push_v = Vector2(0, -1)
		
		if(dir_queue.back() != push_v):
			dir_queue.push_back(push_v)
	elif(Input.is_action_pressed("ui_left")):
		push_v = Vector2(-1, 0)
		
		if(dir_queue.back() != push_v):
			dir_queue.push_back(push_v)
	elif(Input.is_action_pressed("ui_right")):
		push_v = Vector2(1, 0)
		
		if(dir_queue.back() != push_v):
			dir_queue.push_back(push_v)
	else:
		pass
	
	#snake always moves
	move_snake()

#this function is used only in Game.gd, because snake can change it's direction
#only when timer stopped(timer in Game.tscn) 
func change_direction(dir_queue):
	#if dir_queue is empty there is no new directions,
	if(dir_queue.size() == 0):
		return
	
	#local variable; 
	#else first_dir is the first element in dir_queue(current direction) 
	var first_dir = dir_queue.pop_front()
	if(first_dir == Vector2(0, 1)):
		
		#if first_dir == -direction, then snake will bump,
		#so then first_dir can't be current direction
		if(direction == Vector2(0, -1)):
			
			#while first element in dir_queue is equal to first_dir, 
			#it popped out
			while(dir_queue.front() == first_dir):
				dir_queue.pop_front()
			
			#after this loop, direction can be changed to the first in dir_queue
			change_direction(dir_queue)
		else:
			
			#Sprite of head for each case of snake moving
			$head/sprite_head_down.visible = true
			$head/sprite_head_left.visible = false
			$head/sprite_head_right.visible = false
			$head/sprite_head_up.visible = false
			
			#the same idea to the head collisions
			$head/Head_collision_down.disabled = false
			$head/Head_collision_right.disabled = true
			$head/Head_collision_up.disabled = true
			$head/Head_colllision_left.disabled = true
			#new direction
			direction = Vector2(0, 1)
	elif(first_dir == Vector2(0, -1)):
		if(direction == Vector2(0, 1)):
			while(dir_queue.front() == first_dir):
				dir_queue.pop_front()
			change_direction(dir_queue)
		else:
			$head/sprite_head_down.visible = false
			$head/sprite_head_left.visible = false
			$head/sprite_head_right.visible = false
			$head/sprite_head_up.visible = true
	
			$head/Head_collision_down.disabled = true
			$head/Head_collision_right.disabled = true
			$head/Head_collision_up.disabled = false
			$head/Head_colllision_left.disabled = true
			direction = Vector2(0, -1)
	elif(first_dir == Vector2(1, 0)):
		if(direction == Vector2(-1, 0)):
			while(dir_queue.front() == first_dir):
				dir_queue.pop_front()
			change_direction(dir_queue)
		else:
			$head/sprite_head_down.visible = false
			$head/sprite_head_left.visible = false
			$head/sprite_head_right.visible = true
			$head/sprite_head_up.visible = false
	
			$head/Head_collision_down.disabled = true
			$head/Head_collision_right.disabled = false
			$head/Head_collision_up.disabled = true
			$head/Head_colllision_left.disabled = true
			direction = Vector2(1, 0)
	elif(first_dir == Vector2(-1, 0)):
		if(direction == Vector2(1, 0)):
			while(dir_queue.front() == first_dir):
				dir_queue.pop_front()
			change_direction(dir_queue)
		else:
			$head/sprite_head_down.visible = false
			$head/sprite_head_left.visible = true
			$head/sprite_head_right.visible = false
			$head/sprite_head_up.visible = false
		
			$head/Head_collision_down.disabled = true
			$head/Head_collision_right.disabled = true
			$head/Head_collision_up.disabled = true
			$head/Head_colllision_left.disabled = false
			direction = Vector2(-1, 0)
	else:
		pass

func move_snake():
	#local bool flag: true when direction is new
	var dir_change = false
	
	#updating directions of previous snake block
	if(prev_tail_dir != direction):
		prev_tail_dir = direction
		dir_change = true
	
	#head position
	var head_pos = get_node("head").position
	
	#changing head position
	get_node("head").position += direction/2
	
	#moving all previous snake blocks
	if(dir_change):
		for i in range(1, get_child_count()):
			get_child(i).add_to_tail(head_pos, direction)

func add_body():
	#new body instance
	var inst = body.instance()
	
	#the last body block
	var prev_body_element = get_child(get_child_count() - 2)
	
	#tail block(it must be removed to add body block, then add tail block)
	var remover = get_child(get_child_count() - 1)
	
	#if previous element is body block constructing it
	#(new body block; read Body.gd)
	if(prev_body_element.name != "head"):
		#current direction of previous block is equal to previous direction
		#of current block
		inst.cur_dir = prev_body_element.cur_dir
		
		#the same to the position array and direction array
		for i in range(prev_body_element.pos_array.size()):
			inst.pos_array.append(prev_body_element.pos_array[i])
			inst.directions.append(prev_body_element.directions[i])
		
		#position of block is previous position of next block + distance between
		#blocks(+ prev_....cur_dir * gap because gap is lower than zero)
		inst.position = prev_body_element.position + prev_body_element.cur_dir * gap
	#if new element is the first body block, it's arrays are empty and direction
	#is equal to head direction
	else:
		inst.cur_dir = direction
		inst.position = prev_body_element.position + gap * direction
	
	#removing tail block
	remove_child(remover)
	
	#each block connected to the function, that destroys each previous block and
	#current block(if snake bumped into it)
	inst.connect("touch", self, "destroy")
	
	#adding body block
	add_child(inst)
	
	#adding tail block
	add_tail()

func destroy(element):
	#local variable;
	#the last snake element(tail)
	var temp_element = get_child(get_child_count() - 1)
	
	#while element is not the last block are being destroyed
	while(temp_element != element):
		remove_child(temp_element)
		temp_element = get_child(get_child_count() - 1)
	
	#removing current block
	remove_child(element)
	
	#adding tail
	add_tail()

#this function is the same as add_body() function; difference only in that this
#function is only for tail blocks
func add_tail():
	var inst = tail.instance()
	var last_body_element = get_child(get_child_count() - 1)
	if(last_body_element.name != "head"):
		inst.cur_dir = last_body_element.cur_dir
		for i in range(last_body_element.pos_array.size()):
			inst.pos_array.append(last_body_element.pos_array[i])
			inst.directions.append(last_body_element.directions[i])
		inst.position = last_body_element.position + last_body_element.cur_dir * gap
	else:
		inst.cur_dir = direction
		inst.position = last_body_element.position + gap * direction
	add_child(inst)
