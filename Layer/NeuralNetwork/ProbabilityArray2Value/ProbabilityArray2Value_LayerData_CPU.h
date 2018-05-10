//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __ProbabilityArray2Value_LAYERDATA_CPU_H__
#define __ProbabilityArray2Value_LAYERDATA_CPU_H__

#include"ProbabilityArray2Value_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ProbabilityArray2Value_LayerData_CPU : public ProbabilityArray2Value_LayerData_Base
	{
		friend class ProbabilityArray2Value_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		ProbabilityArray2Value_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~ProbabilityArray2Value_LayerData_CPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif