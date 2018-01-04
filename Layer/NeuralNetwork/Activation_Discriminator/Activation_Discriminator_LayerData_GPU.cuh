//======================================
// 活性化関数のレイヤーデータ
//======================================
#ifndef __Activation_Discriminator_LAYERDATA_GPU_H__
#define __Activation_Discriminator_LAYERDATA_GPU_H__

#include"Activation_Discriminator_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Activation_Discriminator_LayerData_GPU : public Activation_Discriminator_LayerData_Base
	{
		friend class Activation_Discriminator_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Activation_Discriminator_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Activation_Discriminator_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** レイヤーを作成する.
			ホストメモリで結果を返すレイヤー.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要 */
		ILayerBase* CreateLayer_host(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** レイヤーを作成する.
			デバイスメモリで結果を返すレイヤー.CPU版では未使用.
			@param	guid	新規生成するレイヤーのGUID.
			@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要 */
		ILayerBase* CreateLayer_device(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif