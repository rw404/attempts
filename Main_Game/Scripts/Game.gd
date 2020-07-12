extends Node2D

onready var food = preload("res://Scenes/Food.tscn")

#Score&HighScore
var score = 0
export var highscore = 0

func _physics_process(delta):
	score = get_node("Snake").get_child_count() - 2
	highscore = max(highscore, score)
	$Score.text = "Score: " + str(score)
	$HighScore.text = "Highscore: " + str(highscore)

func _ready():
	add_food()

func add_food():
	var inst = food.instance()
	inst.position = Vector2(get_random(309, 11), get_random(169, 11))
	inst.connect("eaten", self, "spawn_and_grow")
	add_child(inst)

func spawn_and_grow():
	score += 1
	add_food()
	get_node("Snake").add_tail()

func get_random(_max, _min):
	randomize()
	var x = (randi() % _max + _min) % _max
	if(x < _min):
		x += _min
	return x


func _on_Area2D_area_entered(area):
	if(area.name == "head"):
		get_node("Snake/head").position = get_node("Snake").default_position
		get_node("Snake").direction = Vector2(1, 0)
		
		
		$Snake/head/sprite_head_down.visible = false
		$Snake/head/sprite_head_left.visible = false
		$Snake/head/sprite_head_right.visible = true
		$Snake/head/sprite_head_up.visible = false
		
		$Snake/head/Head_collision_down.disabled = true
		$Snake/head/Head_collision_right.disabled = false
		$Snake/head/Head_collision_up.disabled = true
		$Snake/head/Head_colllision_left.disabled = true
		
		remove_child(get_child(get_child_count() - 1))
		
		add_food()
		
		get_node("Snake").destroy(get_node("Snake").get_child(1))
		#get_tree().reload_current_scene()
