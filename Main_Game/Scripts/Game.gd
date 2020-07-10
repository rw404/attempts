extends Node2D

onready var food = preload("res://Scenes/Food.tscn")

func _ready():
	add_food()

func add_food():
	var inst = food.instance()
	inst.position = Vector2(get_random(320, 0), get_random(180, 0))
	inst.connect("eaten", self, "spawn_and_grow")
	add_child(inst)

func spawn_and_grow():
	add_food()
	get_node("Snake").add_tail()

func get_random(_max, _min):
	randomize()
	var x = randi() % _max + _min
	return x
