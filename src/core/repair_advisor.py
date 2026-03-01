"""
Repair Advisor — Material Calculator & Cost Estimator
Recommends repair methods, calculates required materials (kg),
and estimates costs (IDR) based on pothole volume and severity.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RepairStep:
    """A single step in the repair procedure"""
    order: int
    action: str
    description: str
    duration_minutes: int


@dataclass
class RepairRecommendation:
    """Complete repair recommendation"""
    method: str                     # Repair method name
    method_id: str                  # Indonesian name
    material_name: str              # Primary material
    material_kg: float              # Material needed (kg)
    material_cost_idr: float        # Material cost in IDR
    labor_cost_idr: float           # Estimated labor cost
    total_cost_idr: float           # Total estimated cost
    tools_needed: List[str]         # Required tools
    steps: List[RepairStep]         # Repair procedure
    estimated_time_hours: float     # Total repair time
    durability_months: int          # Expected lifespan of repair
    notes: str                      # Additional notes


class RepairAdvisor:
    """
    Calculate repair recommendations based on pothole measurements.
    
    Supports three repair methods (FHWA-standardized):
    1. Throw-and-Roll — quick temporary fix for small potholes
    2. Semi-Permanent — proper patching for medium potholes
    3. Full-Depth — complete reconstruction for severe damage
    
    Material costs are in IDR (Indonesian Rupiah), calibrated
    for 2024-2025 market prices.
    """
    
    # Material specifications
    MATERIALS = {
        'cold_mix_asphalt': {
            'name': 'Aspal Dingin (Cold Mix)',
            'density_kg_per_cm3': 0.0021,     # ~2.1 g/cm³ → kg/cm³
            'price_per_kg_idr': 8000,
            'compaction_factor': 1.25,          # 25% extra for compaction loss
        },
        'hot_mix_asphalt': {
            'name': 'Aspal Panas (Hot Mix / AC-WC)',
            'density_kg_per_cm3': 0.0023,     # ~2.3 g/cm³
            'price_per_kg_idr': 6000,
            'compaction_factor': 1.15,
        },
        'concrete_patch': {
            'name': 'Beton Tambal (Concrete Patch)',
            'density_kg_per_cm3': 0.0024,     # ~2.4 g/cm³
            'price_per_kg_idr': 5000,
            'compaction_factor': 1.10,
        },
    }
    
    # Labor cost estimate (per pothole)
    LABOR_RATES = {
        'throw_and_roll': 50000,       # IDR per pothole
        'semi_permanent': 150000,
        'full_depth': 500000,
    }
    
    def recommend(
        self,
        volume_cm3: float,
        depth_cm: float,
        area_cm2: float,
        severity_level: str = 'MEDIUM',
        surface_type: str = 'asphalt',
    ) -> RepairRecommendation:
        """
        Generate complete repair recommendation.
        
        Args:
            volume_cm3: Pothole volume in cm³
            depth_cm: Average depth in cm
            area_cm2: Surface area in cm²
            severity_level: From SeverityClassifier (LOW/MEDIUM/HIGH/CRITICAL)
            surface_type: Road surface type ('asphalt' or 'concrete')
            
        Returns:
            RepairRecommendation with full details
        """
        # Select repair method based on severity
        method_key = self._select_method(severity_level, depth_cm, volume_cm3)
        
        # Select material based on method and surface type
        material_key = self._select_material(method_key, surface_type)
        material = self.MATERIALS[material_key]
        
        # Calculate material quantity
        # Volume × density × compaction factor = material needed
        raw_kg = volume_cm3 * material['density_kg_per_cm3']
        total_kg = raw_kg * material['compaction_factor']
        
        # Calculate costs
        material_cost = total_kg * material['price_per_kg_idr']
        labor_cost = self.LABOR_RATES[method_key]
        total_cost = material_cost + labor_cost
        
        # Get repair procedure
        steps = self._get_repair_steps(method_key)
        tools = self._get_tools(method_key)
        
        # Estimate repair time
        time_hours = sum(s.duration_minutes for s in steps) / 60.0
        
        # Durability estimate
        durability = self._estimate_durability(method_key)
        
        # Method names
        method_names = {
            'throw_and_roll': ('Throw-and-Roll', 'Lempar dan Gilas'),
            'semi_permanent': ('Semi-Permanent Patch', 'Tambal Semi-Permanen'),
            'full_depth': ('Full-Depth Repair', 'Perbaikan Kedalaman Penuh'),
        }
        method_en, method_id = method_names[method_key]
        
        # Notes
        notes = self._generate_notes(method_key, total_kg, depth_cm)
        
        return RepairRecommendation(
            method=method_en,
            method_id=method_id,
            material_name=material['name'],
            material_kg=round(total_kg, 2),
            material_cost_idr=round(material_cost),
            labor_cost_idr=labor_cost,
            total_cost_idr=round(total_cost),
            tools_needed=tools,
            steps=steps,
            estimated_time_hours=round(time_hours, 1),
            durability_months=durability,
            notes=notes,
        )
    
    def _select_method(self, severity: str, depth_cm: float, volume_cm3: float) -> str:
        """Select repair method based on severity and dimensions"""
        if severity == 'LOW' or (depth_cm <= 2.5 and volume_cm3 <= 500):
            return 'throw_and_roll'
        elif severity in ('MEDIUM', 'HIGH') or depth_cm <= 10:
            return 'semi_permanent'
        else:
            return 'full_depth'
    
    def _select_material(self, method: str, surface_type: str) -> str:
        """Select material based on method and surface"""
        if surface_type == 'concrete':
            return 'concrete_patch'
        elif method == 'throw_and_roll':
            return 'cold_mix_asphalt'  # Cold mix for quick fixes
        else:
            return 'hot_mix_asphalt'   # Hot mix for permanent repairs
    
    def _get_repair_steps(self, method: str) -> List[RepairStep]:
        """Get procedural steps for repair method"""
        if method == 'throw_and_roll':
            return [
                RepairStep(1, "Bersihkan", "Buang debris dan air dari lubang", 5),
                RepairStep(2, "Isi Material", "Tuang cold mix asphalt ke lubang", 5),
                RepairStep(3, "Ratakan", "Ratakan permukaan dengan sekop", 3),
                RepairStep(4, "Padatkan", "Gilas dengan ban kendaraan 2-3 kali", 5),
            ]
        elif method == 'semi_permanent':
            return [
                RepairStep(1, "Bersihkan", "Buang semua debris, air, dan material lepas", 10),
                RepairStep(2, "Potong Tepi", "Buat tepi vertikal/kotak di sekitar lubang", 15),
                RepairStep(3, "Tack Coat", "Aplikasikan emulsi aspal di dasar dan tepi", 5),
                RepairStep(4, "Isi Material", "Tuang hot mix asphalt secara berlapis", 10),
                RepairStep(5, "Padatkan", "Padatkan dengan vibratory compactor", 10),
                RepairStep(6, "Level", "Periksa kerataan permukaan", 5),
            ]
        else:  # full_depth
            return [
                RepairStep(1, "Marking", "Tandai area perbaikan lebih besar dari lubang", 10),
                RepairStep(2, "Potong", "Potong aspal lama dengan saw cutter", 30),
                RepairStep(3, "Buang", "Angkat dan buang material lama", 20),
                RepairStep(4, "Base Prep", "Siapkan dan padatkan base course", 30),
                RepairStep(5, "Tack Coat", "Aplikasikan emulsi aspal pada semua permukaan", 10),
                RepairStep(6, "Paving", "Tuang hot mix asphalt berlapis-lapis", 20),
                RepairStep(7, "Compact", "Padatkan dengan roller compactor", 15),
                RepairStep(8, "Finishing", "Segel tepi dan periksa kerataan", 15),
            ]
    
    def _get_tools(self, method: str) -> List[str]:
        """Get required tools for repair method"""
        if method == 'throw_and_roll':
            return ["Sekop", "Sapu", "Cold mix asphalt"]
        elif method == 'semi_permanent':
            return ["Sekop", "Sapu", "Pahat/chisel", "Emulsi aspal", 
                    "Hot mix asphalt", "Vibratory plate compactor"]
        else:
            return ["Saw cutter", "Jackhammer", "Sekop", "Dump truck",
                    "Roller compactor", "Emulsi aspal", "Hot mix asphalt",
                    "Base course material", "Leveling tools"]
    
    def _estimate_durability(self, method: str) -> int:
        """Estimated repair lifespan in months"""
        return {
            'throw_and_roll': 3,      # Temporary: 3 months
            'semi_permanent': 18,     # Semi-permanent: 1.5 years
            'full_depth': 60,         # Full repair: 5 years
        }[method]
    
    def _generate_notes(self, method: str, material_kg: float, depth_cm: float) -> str:
        """Generate contextual notes"""
        notes = []
        
        if method == 'throw_and_roll':
            notes.append("⚡ Perbaikan sementara — perlu ditinjaklanjuti dalam 3 bulan.")
        
        if material_kg > 50:
            notes.append(f"📦 Material {material_kg:.1f} kg — perlu kendaraan pengangkut.")
        
        if depth_cm > 10:
            notes.append("⚠️ Kedalaman >10cm — periksa base layer dan drainase.")
        
        return " ".join(notes) if notes else "Perbaikan standar dapat dilakukan."
    
    def format_cost_idr(self, amount: float) -> str:
        """Format IDR amount with thousands separator"""
        return f"Rp {amount:,.0f}".replace(",", ".")


if __name__ == "__main__":
    advisor = RepairAdvisor()
    
    # Test cases matching severity test cases
    test_cases = [
        {"volume_cm3": 150,   "depth_cm": 1.5, "area_cm2": 100,  "severity_level": "LOW"},
        {"volume_cm3": 1600,  "depth_cm": 4.0, "area_cm2": 400,  "severity_level": "MEDIUM"},
        {"volume_cm3": 9000,  "depth_cm": 7.5, "area_cm2": 1200, "severity_level": "HIGH"},
        {"volume_cm3": 45000, "depth_cm": 15,  "area_cm2": 3000, "severity_level": "CRITICAL"},
    ]
    
    print("=" * 70)
    print("REPAIR RECOMMENDATION TEST")
    print("=" * 70)
    
    for tc in test_cases:
        rec = advisor.recommend(**tc)
        print(f"\n  Severity:  {tc['severity_level']}")
        print(f"  Method:    {rec.method} ({rec.method_id})")
        print(f"  Material:  {rec.material_name}")
        print(f"  Quantity:  {rec.material_kg} kg")
        print(f"  Cost:      {advisor.format_cost_idr(rec.total_cost_idr)}")
        print(f"  Time:      {rec.estimated_time_hours} hours")
        print(f"  Durability: {rec.durability_months} months")
        if rec.notes:
            print(f"  Notes:     {rec.notes}")
    
    print(f"\n{'='*70}")
