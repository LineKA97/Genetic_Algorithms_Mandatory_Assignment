
import sys
from datetime import datetime
from datetime import timedelta
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

data = pd.read_csv('TSPcities1000.txt',sep='\s+',header=None)
data = pd.DataFrame(data)

x = data[1]
y = data[2]

plt.plot(x, y,'r.')
plt.show()

def createRandomRoute():
    tour = [[i] for i in range(10)]
    random.shuffle(tour)
    return tour

# plot the tour - Adjust range 0..len, if you want to plot only a part of the tour.
def plotCityRoute(route):
    for i in range(0, len(route)):
        plt.plot(x[i:i + 2], y[i:i + 2], 'ro-')
    plt.show()

# calculate distance between cities
def distancebetweenCities(city1x, city1y, city2x, city2y):
    xDistance = abs(city1x-city2x)
    yDistance = abs(city1y-city2y)
    distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
    return distance


#Random population set
initRouts = []
for i in range(0, 15):
    initRouts.append(createRandomRoute())
print('init routes: ', initRouts)

#Fitness
#Calculate distance of initRouts
#returns an array of distance of route in index [0] and an array of city names in [1]
def calculatedistance():
    distance = []
    for i in range(0, len(initRouts)):
        route = initRouts[i]
        total = 0
        for t in range(0, len(route) - 1):
            total += distancebetweenCities(x[route[t][0]], y[route[t][0]], x[route[t+1][0]], y[route[t+1][0]])
        distance.append([total, route])
    return distance


#print('distance: ', calculatedistance())

#Sort distance from calculateddistance() by lowest distance in index [0]
#Evaluering af fitness
def sortByLowestDistance(dis):
   return sorted(dis, key=lambda d: d[0])

sortedRoutes = sortByLowestDistance(calculatedistance())
print("Sorted: ", sortedRoutes)

#Take 50% of the 15 sortedroutes, where we get the 7 shortest routes.
#We don't want decimals, so we use floor to get even numbers.
breeders = sortedRoutes[:math.floor(len(sortedRoutes) * 0.50)]

#Shows number of breeders + the array of breeders
print("Breeders: (" + str(len(breeders)) + ") \n", breeders)


# generer random routs, f.eks. 1000
# fit funktion -> brug hans snippest
# child = complet route, can't have the same city twice
#Zip de to parents, men sørg for at parent 2 ikke flyder child med de samme byer som parent 1


#cross over operation
def breedchildren(breeders):
    children = []
    for i in range(len(breeders) - 1):
        parent1 = breeders[i][1]
        parent2 = breeders[i + 1][1]
        #print("parent1: ", parent1)
        #print("parent2: ", parent2)

        child = parent1[: len(parent1) // 2]

        i = 0
        while len(child) < len(parent2):
            if parent2[i] not in child:
                child.append(parent2[i])
            i += 1

        print("Child:  \n", child)
        children.append(child)
    return children


#Mutation
def mutate(route_to_mut):
    # 50% chance of being mutated
    k_mut_prob = 0.90

    if random.random() < k_mut_prob:
        route_pos1 = random.randint(0, len(route_to_mut) - 1)
        route_pos2 = random.randint(0, len(route_to_mut) - 1)
        if route_pos1 != route_pos2:
                # Swap
                city1 = route_to_mut[route_pos1]
                city2 = route_to_mut[route_pos2]

                route_to_mut[route_pos1] = city2
                route_to_mut[route_pos2] = city1

    return route_to_mut

print("children: ", breedchildren(breeders))

children = breedchildren(breeders)
print("children: ", children)

route_to_mut = breedchildren(breeders)
print("non mutated: \n", route_to_mut)

for i in range(len(route_to_mut)):
    route_to_mut[i] = mutate(route_to_mut[i])
print("mutated: \n", route_to_mut)

#Main
startTime = datetime.now()


tour = createRandomRoute()
print('Tour: ', tour)
plotCityRoute(tour)

# distance between city number 100 and city number 105
dist= distancebetweenCities(x[2], y[2], x[5], y[5])
print('Distance, % target: ', dist)

generateRouteList = []
for i in range(100):
    generateRouteList.append(createRandomRoute())

shortestDistance = sys.maxsize
best_score_progress = []  # Tracks progress
numOfGenerations = 100

for gen in range(numOfGenerations):

    #Fitness
    routeList = []
    for i in range(len(generateRouteList)):
        route = generateRouteList[i]
        total = 0
        for j in range(len(route) - 1):
            total += distancebetweenCities(x[route[j][0]], y[route[j][0]], x[route[j+1][0]], y[route[j+1][0]])
        routeList.append([total, route])

    sortedRoutesList = sortByLowestDistance(routeList)
    print("sortedRoutesList: ", sortedRoutesList)
    if sortedRoutesList[0][0] < shortestDistance:
        shortestDistance = sortedRoutesList[0][1]
        best_score_progress.append(shortestDistance)

    theBest = sortedRoutesList[: math.floor(len(routeList) * 0.50)]
    print("the best: ", theBest)

    children = breedchildren(generateRouteList)
    children = mutate(children)
    print("children: ", children)

    population = children

    for i in range(len(theBest)):
        population.append(generateRouteList[theBest][i][1])

    i = len(theBest)
    while len(population) < len(routeList):
        population.append((generateRouteList[sortedRoutesList][i][1]))
        i += 1

        generateRouteList = population

endTime = datetime.now()
executionTime = endTime - startTime
executionTime = timedelta(seconds = round(executionTime.total_seconds()))

print("Finsihed!")
print("Shortest route distance: ", shortestDistance)

plt.plot(best_score_progress)
plt.show()

''''
# replace with your own calculations
fitness_gen0 = 1000 # replace with your value
print('Starting best score, % target: ', fitness_gen0)

best_score = fitness_gen0
# Add starting best score to progress tracker
best_score_progress.append(best_score)

# Here comes your GA program...
best_score = 980
best_score_progress.append(best_score)
best_score = 960
best_score_progress.append(best_score)


# GA has completed required generation
print('End best score, % target: ', best_score)

plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best Fitness - route length - in Generation')
plt.show()
'''