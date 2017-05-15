//======================================
// 活性化関数のレイヤーデータ
//======================================
#ifndef __ACTIVATION_LAYERDATA_CPU_H__
#define __ACTIVATION_LAYERDATA_CPU_H__

#include"Activation_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Activation_LayerData_CPU : public Activation_LayerData_Base
	{
		friend class Activation_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Activation_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Activation_LayerData_CPU();


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