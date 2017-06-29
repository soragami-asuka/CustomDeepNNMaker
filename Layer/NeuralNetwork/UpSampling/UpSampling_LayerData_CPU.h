//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __UpSampling_LAYERDATA_CPU_H__
#define __UpSampling_LAYERDATA_CPU_H__

#include"UpSampling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class UpSampling_LayerData_CPU : public UpSampling_LayerData_Base
	{
		friend class UpSampling_CPU;

	private:
		// �{��
		std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< �e�j���[�����̌W��<�j���[������, ���͐�> */
		std::vector<NEURON_TYPE>				lpBias;				/**< �j���[�����̃o�C�A�X<�j���[������> */

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		UpSampling_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~UpSampling_LayerData_CPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif