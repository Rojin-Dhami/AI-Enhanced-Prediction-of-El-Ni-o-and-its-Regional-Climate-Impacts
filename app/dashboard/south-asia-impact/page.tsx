import CompositeImpactTable from "@/components/dashboard/CompositeImpactTable";
import KeyInsights from "@/components/dashboard/KeyInsights";

export default function SouthAsiaImpactPage() {
  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CompositeImpactTable />
        <KeyInsights />
      </div>
    </>
  );
}
