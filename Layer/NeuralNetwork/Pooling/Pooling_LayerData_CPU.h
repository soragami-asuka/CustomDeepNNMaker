//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __POOLING_LAYERDATA_CPU_H__
#define __POOLING_LAYERDATA_CPU_H__

#include"Pooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Pooling_LayerData_CPU : public Pooling_LayerData_Base
	{
		friend class Pooling_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Pooling_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Pooling_LayerData_CPU();


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