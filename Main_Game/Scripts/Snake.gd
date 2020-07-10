extends Node2D

var direction = Vector2(1, 0)
onready var tail = preload("res://Scenes/Tail.tscn")
onready var last_tail = preload("res://Scenes/Last_Tail.tscn")
const gap = -8
var next_tail_dir = Vector2(1, 0)
var prev_tail_dir = Vector2(1, 0)

func _ready():
	add_last_tail()

func _physics_process(delta):
	if(Input.is_action_pressed("ui_down")):
		direction = Vector2(0, 1)
		pass
	elif(Input.is_action_pressed("ui_up")):
		direction = Vector2(0, -1)
		pass
	elif(Input.is_action_pressed("ui_left")):
		direction = Vector2(-1, 0)
		pass
	elif(Input.is_action_pressed("ui_right")):
		direction = Vector2(1, 0)
		pass
	else:
		#direction = Vector2.ZERO
		pass
		
	move_snake()

func move_snake():
	var dir_change = false
	if(prev_tail_dir != direction):
		prev_tail_dir = direction
		dir_change = true
	var head_pos = get_node("head").position
	get_node("head").position += direction/2
	
	if(dir_change):
		for i in range(1, get_child_count()):
			get_child(i).add_to_tail(head_pos, direction)

func add_tail():
	var inst = tail.instance()
	var prev_tail = get_child(get_child_count() - 2)
	var remover = get_child(get_child_count() - 1)
	if(prev_tail.name != "head"):
		inst.cur_dir = prev_tail.cur_dir
		for i in range(prev_tail.pos_array.size()):
			inst.pos_array.append(prev_tail.pos_array[i])
			inst.directions.append(prev_tail.directions[i])
		inst.position = prev_tail.position + prev_tail.cur_dir * gap
	else:
		inst.cur_dir = direction
		inst.position = prev_tail.position + gap * direction
	remove_child(remover)
	inst.connect("touch", self, "destroy")
	add_child(inst)
	add_last_tail()

func destroy(element):
	var temp_element = get_child(get_child_count() - 1)
	while(temp_element != element):
		remove_child(temp_element)
		temp_element = get_child(get_child_count() - 1)
	add_last_tail()

func add_last_tail():
	var inst = last_tail.instance()
	var prev_tail = get_child(get_child_count() - 1)
	if(prev_tail.name != "head"):
		inst.cur_dir = prev_tail.cur_dir
		for i in range(prev_tail.pos_array.size()):
			inst.pos_array.append(prev_tail.pos_array[i])
			inst.directions.append(prev_tail.directions[i])
		inst.position = prev_tail.position + prev_tail.cur_dir * gap
	else:
		inst.cur_dir = direction
		inst.position = prev_tail.position + gap * direction
	add_child(inst)
