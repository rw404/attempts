extends Node2D

onready var food = preload("res://Scenes/Food.tscn")


#var score = 0

#Score
#func _physics_process(delta):
#	$RichTextLabel.text = "Score: " + str(score)


func _ready():
	add_food()

func add_food():
	var inst = food.instance()
	inst.position = Vector2(get_random(309, 11), get_random(169, 11))
	inst.connect("eaten", self, "spawn_and_grow")
	add_child(inst)

func spawn_and_grow():
	add_food()
	get_node("Snake").add_tail()
	#score += 1

func get_random(_max, _min):
	randomize()
	var x = (randi() % _max + _min) % _max
	if(x < _min):
		x += _min
	return x


func _on_Area2D_area_entered(area):
	if(area.name == "head"):
		get_tree().reload_current_scene()
