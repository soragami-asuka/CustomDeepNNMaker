//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __CONVOLUTION_LAYERDATA_CPU_H__
#define __CONVOLUTION_LAYERDATA_CPU_H__

#include"Convolution_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Convolution_LayerData_CPU : public Convolution_LayerData_Base
	{
		friend class Convolution_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Convolution_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Convolution_LayerData_CPU();


		//===========================
		// ������
		//===========================
	public:
		using Convolution_LayerData_Base::Initialize;

		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:

		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
	public:
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif