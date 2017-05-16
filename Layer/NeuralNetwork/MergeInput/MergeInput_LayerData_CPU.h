//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __MergeInput_LAYERDATA_CPU_H__
#define __MergeInput_LAYERDATA_CPU_H__

#include"MergeInput_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeInput_LayerData_CPU : public MergeInput_LayerData_Base
	{
		friend class MergeInput_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		MergeInput_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~MergeInput_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif