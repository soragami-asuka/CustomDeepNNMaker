//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __Residual_LAYERDATA_CPU_H__
#define __Residual_LAYERDATA_CPU_H__

#include"Residual_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Residual_LayerData_CPU : public Residual_LayerData_Base
	{
		friend class Residual_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Residual_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Residual_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif