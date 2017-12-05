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
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要 */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif