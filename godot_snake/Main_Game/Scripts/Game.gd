extends Node2D

#Global variables:
	#variable for food instances:
onready var food = preload("res://Scenes/Food.tscn")
	#variable to start, stop, signal the timer
onready var timer = get_node("Timer")
	#Score
var score = 0
	#Highscore
var highscore = 0

func _physics_process(delta):
	#each time score is equal to the number of snake blocks minus head and tail
	score = get_node("Snake").get_child_count() - 2
	
	#highscore is the max of previous highscore and current score
	highscore = max(highscore, score)
	
	#printing score and highscore on the field
	$Score.text = "Score: " + str(score)
	$HighScore.text = "Highscore: " + str(highscore)

func _ready():
	#when this scene is loadaed timer starts and food adds
	timer.start()
	add_food()

func add_food():
	#inst is a food object
	var inst = food.instance()
	
	#position of food object is random, but it must be on field(not out)
	inst.position = Vector2(get_random(309, 11), get_random(169, 11))
	
	#when food is eaten, snake grows and new food adds
	inst.connect("eaten", self, "spawn_and_grow")
	add_child(inst)

func spawn_and_grow():
	add_food()
	get_node("Snake").add_body()

func get_random(_max, _min):
	#random x, which _max > x >= _min
	randomize()
	var x = (randi() % _max + _min) % _max
	if(x < _min):
		x += _min
	return x

func _on_Area2D_area_entered(area):
	#if snake bumped into the walls game restarts, but highscore is saved
	if(area.name == "head"):
		#snake position and direction is equal to the start position and 
		#direction
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
		
		#removing food object
		remove_child(get_child(get_child_count() - 1))
		
		#adding new food object(because it helps to locate food object with new
		#position)
		add_food()
		
		#destroying all snake
		get_node("Snake").destroy(get_node("Snake").get_child(1))

func _on_Timer_timeout():
	#when timer is stopped, snake can change direction and timer restarts
	get_node("Snake").change_direction(get_node("Snake").dir_queue)
	timer.start()
