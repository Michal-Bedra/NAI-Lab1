import random
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
Program steruje mocą ssania odkurzacza autonomicznego w zależności od powierzchni, stopnia
zabrudzenia, oraz poziomu dostępnej mocy baterii.
Filip Patuła s28615, Michał Bedra s28854
"""

#stałe do regulatora mocy ssania
DEFAULT_BATTERY_LEVEL = 100.
ENVIRONMENT_LIMIT = 1000
MAX_ROUGHNESS = 10
MAX_DIRTINESS = 20

#stałe do animacji środowiska
X_LIMIT = 40
Y_LIMIT = 6
X_START = -2
WHEEL_ONE_START = -1.6
WHEEL_TWO_START = -0.2
CURRENT_ENVIRONMENT_PAGE = 0


def get_dirtiness_label(dirtiness_value):
    """ Ustala nazwę dla wartości zabrudzenia na podstawie funkcji przynależności
        Parametry:
        dirtiness_value: wartość zabrudzenia
        Zwraca:
        nazwa przypisana do wartości zabrudzenia
    """
    clean = fuzz.interp_membership(dirtiness.universe, dirtiness['clean'].mf, dirtiness_value)
    dirty = fuzz.interp_membership(dirtiness.universe, dirtiness['dirty'].mf, dirtiness_value)
    very_dirty = fuzz.interp_membership(dirtiness.universe, dirtiness['very_dirty'].mf, dirtiness_value)
    return 'clean' if clean > dirty and clean > very_dirty else 'dirty' if dirty > very_dirty else 'very_dirty'

def get_roughness_label(roughness_value):
    """ Ustala nazwę dla wartości chropowatości na podstawie funkcji przynależności
        Parametry:
        dirtiness_value: wartość chropowatości
        Zwraca:
        nazwa przypisana do wartości chropowatości
    """
    smooth = fuzz.interp_membership(roughness.universe, roughness['smooth'].mf, roughness_value)
    medium = fuzz.interp_membership(roughness.universe, roughness['medium'].mf, roughness_value)
    rough = fuzz.interp_membership(roughness.universe, roughness['rough'].mf, roughness_value)
    return 'smooth' if smooth > medium and smooth > rough else 'medium' if medium > rough else 'rough'

def surface_factory(roughness_value, dirtiness_balue, position):
    """ Zwraca kawałek powierzchni do animacji środowiska
        Parametry:
        dirtiness_value: wartość zabrudzenia
        dirtiness_value: wartość chropowatości
        position: pozycja x powierzchni do animacji
        Zwraca:
        wygenerowana na podstawie parametrów powierzchnia
    """
    get_color = lambda x: 'lavender' if x == 'clean' else 'gray' if x == 'dirty' else 'brown'
    create_smooth_surface = lambda x, color : patches.Rectangle((x, 0), 1, 1, color=color, alpha=0.5)
    create_medium_surface = lambda x, color : patches.Ellipse((x + 0.5, 0.5), width=1, height=1, color=color, alpha=0.5)
    create_rough_surface = lambda x, color : patches.Polygon([[x, 0], [x+1, 0],  [x+0.5, 1]], closed=True, color=color, alpha=0.5)

    create_surface = lambda x, roughness_val, dirtiness_val, color_func : create_smooth_surface(x, color_func(dirtiness_val)) if roughness_val == 'smooth' else create_medium_surface(x, color_func(dirtiness_val)) if roughness_val == 'medium' else create_rough_surface(x, color_func(dirtiness_val))
    return create_surface(position, roughness_value, dirtiness_balue, get_color)


class DirtyEnvironmentAnimation:
    """ Animacja dla środowiska wirtualnego regulatora mocy ssania odkurzacza"""
    def __init__(self, dirty_environment=None):
        """ Ustawia podstawowe pola (środowisko, osie, rysowane obiekty) w klasie potrzebne do animowania symulacji i stan początkowy animacji (tło, powierzchnia)
        Parametry:
        self
        dirty_environment: środowisko do symulacji
        Zwraca:
        None
        """
        # podstawowe pola
        self.page = 0
        self.environment = dirty_environment
        self.animated_environment = None

        # podstawowe parametry dla pola rysowania
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(0, X_LIMIT)
        self.ax.set_ylim(0, Y_LIMIT)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # tło
        self.wall = patches.Rectangle((0, 1), X_LIMIT, 5, color='skyblue')
        self.ax.add_patch(self.wall)

        # części odkurzacza
        self.suction_text = self.ax.text(2, 4, 'Suction: 0', fontsize=10, color='black')
        body = patches.Rectangle((X_START, 1), 2, 1, color='lightgray')
        handle = patches.Rectangle((X_START, 2), 0.2, 2, color='gray')
        wheel1 = patches.Circle((WHEEL_ONE_START, 1), 0.2, color='black')
        wheel2 = patches.Circle((WHEEL_TWO_START, 1), 0.2, color='black')
        self.parts = {'body': body, 'handle': handle, 'wheel1':wheel1, 'wheel2':wheel2}

        for part in self.parts.values():
            self.ax.add_patch(part)

        #początkowa powierzchnia
        environment_variables = environment.get_environment_variables(X_LIMIT, self.page)
        self.page += 1
        for position, variables in enumerate(environment_variables):
            surface_patch = surface_factory(get_roughness_label(variables['roughness']), get_dirtiness_label(variables['dirtiness']), position)
            self.ax.add_patch(surface_patch)

    def animate(self, frame):
        """ Generuje obiekty na kolejnej klatce animacji do symulacji regulatora mocy ssania odkurzacza, w przypadku przekroczenia obszaru rysowania generuje dalszą powierzchnię, zatrzymuje się kiedy bateria odkurzacza wynosi 0
        Parametry:
        self
        frame: numer klatki
        Zwraca:
        None
        """
        environment_phase_values = self.environment.increment_environment_phase()
        if environment_phase_values['battery'] < 0:
            self.ax.text(X_LIMIT-10, 4, "Out of battery!", fontsize=10, color='red', fontweight='bold')
            self.animated_environment.event_source.stop()
            return []
        redraw_objects = list(self.parts.values())
        self.suction_text.set_text(f"Suction: {round(environment_phase_values['suction'], 1)}")
        redraw_objects.append(self.suction_text)
        body = self.parts['body']
        handle = self.parts['handle']
        wheel1 = self.parts['wheel1']
        wheel2 = self.parts['wheel2']
        dx = 1
        body.set_x(body.get_x() + dx)
        handle.set_x(handle.get_x() + dx)
        wheel1.center = (wheel1.center[0] + dx, wheel1.center[1])
        wheel2.center = (wheel2.center[0] + dx, wheel2.center[1])

        # Reset when off-screen
        if body.get_x() + 2 > X_LIMIT:
            body.set_x(-1)
            handle.set_x(-1)
            wheel1.center = (-0.6, 1)
            wheel2.center = (0.8, 1)
            self.ax.clear()
            self.ax.set_xlim(0, X_LIMIT)
            self.ax.set_ylim(0, Y_LIMIT)
            self.ax.axis('off')
            self.ax.add_patch(self.wall)
            redraw_objects.append(self.wall)
            self.suction_text = self.ax.text(2, 4, f"Suction: {round(environment_phase_values['suction'], 1)}", fontsize=10, color='black')
            environment_variables = self.environment.get_environment_variables(X_LIMIT, self.page)
            self.page += 1
            for position, variables in enumerate(environment_variables):
                surface_patch = surface_factory(get_roughness_label(variables['roughness']),
                                                get_dirtiness_label(variables['dirtiness']),
                                                position)
                self.ax.add_patch(surface_patch)
                redraw_objects.append(surface_patch)
            for p in self.parts.values():
                self.ax.add_patch(p)
        return redraw_objects

    def run_animation(self):
        """
        Tworzy obiekt animacji i uruchamia animację
        Parametry:
        self
        Zwraca:
        None
        """
        self.animated_environment = animation.FuncAnimation(self.figure, self.animate, frames=60, interval=100, blit=False)

def prepare_environment(limit):
    """
    Generuje dane dla środowiska do symulacji regulatora mocy ssania odkurzacza
    Parametry:
    limit: ilość danych do wygenerowania
    Zwraca:
    listę słowników z wartościami chropowatości i zabrudzenia powierzchni
    """
    environment_variables = []
    for _ in range(limit):
        environment_variables.append({'roughness':random.uniform(0.1, MAX_ROUGHNESS), 'dirtiness':random.uniform(0.1, MAX_DIRTINESS)})
    return environment_variables


class DirtyEnvironment:
    """ Srodowisko dla wirtualnego regulatora mocy ssania odkurzacza"""

    def __init__(self, control_simulation, battery_level=100., environment_limit=1000):
        """ Inicjalizuje podstawowy stan środowisko dla regulatora mocy ssania odkurzacza
            Parametry:
            control_simulation: skonfigurowane środowisko kontrolne logiki rozmytej z bilioteki skfuzzy
            battery_level: domyślny stan naładowania baterii
            environment_limit: limit ilości stanów dla koljenych faz środowiska
            Zwraca:
            None
        """
        random.seed()
        self.controlSimulation = control_simulation
        self.batteryLevel = battery_level
        self.environmentVariable = prepare_environment(environment_limit)
        self.environmentLimit = environment_limit
        self.currentPhase = 0

    def increment_environment_phase(self):
        """
        Przechodzi do kolejnej fazy środowiska i aktualizuje jego stan, zwraca wartości dla aktualnej fazy symulacji
        Parametry:
        self
        Zwraca:
        słownik z wartościami dla chropowatości, zabrudzenia powierzchni, stanu baterii, mocy ssania i wykorzystania baterii
        """
        phase_values = {}
        if self.batteryLevel <= 0:
            phase_values['battery'] = 0
        else:
            current_environment_variables = self.environmentVariable[self.currentPhase]
            self.controlSimulation.input['roughness'] = current_environment_variables['roughness']
            self.controlSimulation.input['battery'] = self.batteryLevel
            self.controlSimulation.input['dirtiness'] = current_environment_variables['dirtiness']
            self.controlSimulation.compute()
            self.batteryLevel -= self.controlSimulation.output['battery_usage']
            phase_values['roughness'] = current_environment_variables['roughness']
            phase_values['dirtiness'] = current_environment_variables['dirtiness']
            phase_values['battery'] = self.batteryLevel
            phase_values['suction'] = self.controlSimulation.output['suction']
            phase_values['battery_usage'] = self.controlSimulation.output['battery_usage']
        self.currentPhase += 1
        return phase_values

    def get_environment_variables(self, number, page):
        """
        Zwraca wartości dla wybranego zakresu faz
        Parametry:
        self
        number: liczba faz dla których mają zostać zwrócone wartości
        page: mnożnik strony dla wartości
        Zwraca:
        lista słowników z wartościami dla faz
        """
        start = number * page
        stop = start + number
        return self.environmentVariable[start:(stop if stop < self.environmentLimit else self.environmentLimit)]

#  Zmienne i zakresy
roughness = ctrl.Antecedent(np.arange(0, 11, 1), 'roughness')  # chropowatość powierzchni 0=gładka, 10=chropowata
battery = ctrl.Antecedent(np.arange(0, 101, 0.1), 'battery')   # % naładowania baterii
dirtiness = ctrl.Antecedent(np.arange(0, 21, 1), 'dirtiness')  # stopień zabrudzenia
suction = ctrl.Consequent(np.arange(0, 101, 1), 'suction')     # % mocy ssania
batteryUsage = ctrl.Consequent(np.arange(0, 1, 0.1), 'battery_usage')

# Chropowatość 
roughness['smooth'] = fuzz.trimf(roughness.universe, [0, 0, 4]) #gładka
roughness['medium'] = fuzz.trimf(roughness.universe, [3, 6, 8]) #średnia
roughness['rough'] = fuzz.trimf(roughness.universe, [6, 10, 10]) #chropowata

# Bateria
battery['very_low'] = fuzz.trapmf(battery.universe, [0, 0, 14, 16])    #bardzo niska (<~15)
battery['low'] = fuzz.trapmf(battery.universe, [14, 17, 33, 37])    #niska (~15– 35)
battery['high'] = fuzz.trapmf(battery.universe, [33, 38, 100, 100])    #wysoka (>=~35)

# Moc ssania
suction['low'] = fuzz.trapmf(suction.universe, [0, 0, 28, 32]) #niska
suction['medium'] = fuzz.trapmf(suction.universe, [27, 34, 78, 81]) #średnia
suction['high'] = fuzz.trapmf(suction.universe, [77, 81, 100, 100]) #wysoka

# Stopień zabrudzenia
dirtiness['clean'] = fuzz.trimf(dirtiness.universe, [0, 0, 4]) #czysto
dirtiness['dirty'] = fuzz.trimf(dirtiness.universe, [3, 7, 16]) #brudno
dirtiness['very_dirty'] = fuzz.trimf(dirtiness.universe, [10, 20, 20]) #bardzo brudno

# Wykorzystanie baterii
batteryUsage.automf(3, variable_type='quant') #niskie, średnie, wysokie

#zasady dla małej mocy ssania
lowSuctionRule1 = ctrl.Rule(dirtiness['clean'], suction['low'])
lowSuctionRule2 = ctrl.Rule(battery['very_low'], suction['low'])
lowSuctionRule3 = ctrl.Rule(roughness['smooth'] & dirtiness['dirty'], suction['low'])

#zasady dla średniej mocy ssania
mediumSuctionRule1 = ctrl.Rule(battery['high'] & dirtiness['dirty'], suction['medium'])
mediumSuctionRule2 = ctrl.Rule(battery['low'] & dirtiness['dirty'] & roughness['medium'], suction['medium'])
mediumSuctionRule3 = ctrl.Rule(battery['high'] & dirtiness['very_dirty'] & roughness['smooth'], suction['medium'])
mediumSuctionRule4 = ctrl.Rule(battery['low'] & dirtiness['very_dirty'] & roughness['medium'], suction['medium'])
mediumSuctionRule5 = ctrl.Rule(battery['low'] & dirtiness['dirty'] & roughness['rough'], suction['medium'])
mediumSuctionRule6 = ctrl.Rule(battery['low'] & dirtiness['very_dirty'] & roughness['smooth'], suction['medium'])
mediumSuctionRule7 = ctrl.Rule(battery['low'] & dirtiness['very_dirty'] & roughness['rough'], suction['medium'])

#zasady dla wysokiej mocy ssania
highSuctionRule1 = ctrl.Rule(battery['high'] & dirtiness['dirty'] & roughness['rough'], suction['high'])
highSuctionRule2 = ctrl.Rule(battery['high'] & dirtiness['very_dirty'] & roughness['medium'], suction['high'])
highSuctionRule3 = ctrl.Rule(battery['high'] & dirtiness['very_dirty'] & roughness['rough'], suction['high'])

#zasady dla stopnia wykorzystania baterii
batteryUsageRule1 = ctrl.Rule(suction['low'], batteryUsage['low'])
batteryUsageRule2 = ctrl.Rule(suction['medium'], batteryUsage['average'])
batteryUsageRule3 = ctrl.Rule(suction['high'], batteryUsage['high'])

#utworzenie systemu kontroli dla podanych zasad
controlSystem = ctrl.ControlSystem([lowSuctionRule1, lowSuctionRule2, lowSuctionRule3, mediumSuctionRule1, mediumSuctionRule2, mediumSuctionRule3, mediumSuctionRule4, mediumSuctionRule5, mediumSuctionRule6, mediumSuctionRule7, highSuctionRule1, highSuctionRule2, highSuctionRule3, batteryUsageRule1, batteryUsageRule2, batteryUsageRule3])

#utworzenie symulacji dla systemu
simulation = ctrl.ControlSystemSimulation(controlSystem)

#utworzenie środowiska dla symulacji
environment = DirtyEnvironment(control_simulation=simulation, battery_level=DEFAULT_BATTERY_LEVEL, environment_limit=ENVIRONMENT_LIMIT)

#utworzenie animacji dla środowiska symulacji
environment_animation = DirtyEnvironmentAnimation(environment)
#uruchomienie animacji
environment_animation.run_animation()
plt.show()